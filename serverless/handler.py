"""
RunPod Serverless handler for LongCat-Video (text-to-video).

What this file does
-------------------
1. On worker start-up ("cold start") it loads the LongCat-Video model onto the GPU
   ONE time. The model stays in GPU memory for as long as the worker is alive, so
   the 2nd, 3rd, ... requests are much faster than the 1st one.
2. Every time you send a request, `handler()` runs, generates an mp4, and returns it.

Differences vs. the original `run_demo_text_to_video.py`:
  * No `torchrun` / `torch.distributed` / multi-GPU code. A serverless worker is a
    single process on a single GPU, and `init_process_group` would hang there.
  * Prompt and settings come from the request JSON instead of being hard-coded.
  * The result is returned over HTTP (base64 or an S3 link) instead of being written
    to a local file, because a serverless worker's disk disappears after the job.

Environment variables (all optional, set them in the RunPod endpoint UI):
  MODEL_DIR            Where the weights live. Default: auto-detected, normally
                       /runpod-volume/weights/LongCat-Video
  LOW_VRAM             "1" = keep the text encoder in CPU RAM between uses (saves
                       ~11 GB VRAM), "0" = never. Default: auto (on if GPU < 70 GB).
  MAX_RESPONSE_MB      Largest base64 video to return inline. Default 7.
  BUCKET_ENDPOINT_URL  If set (plus BUCKET_ACCESS_KEY_ID / BUCKET_SECRET_ACCESS_KEY)
                       the mp4 is uploaded to your S3 bucket and a link is returned.
"""

import base64
import os
import tempfile
import threading
import time
import traceback

import numpy as np
import PIL.Image
import torch
import runpod
from torchvision.io import write_video
from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.context_parallel import context_parallel_util
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

# --------------------------------------------------------------------------------------
# Defaults. Changing these changes the behaviour of every request that does not
# explicitly override the value.
# --------------------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

# The VAE compresses space by 8 and the DiT patchifies by 2 -> width/height must be
# a multiple of 16. Time is compressed by 4 -> num_frames must be 4*k + 1.
SPATIAL_MULTIPLE = 16
TEMPORAL_MULTIPLE = 4

# Safety rails so a typo in a request cannot burn an hour of GPU time.
MAX_PIXELS = 1280 * 720
MAX_FRAMES = 181
MAX_STEPS = 60

MODEL_DIR_CANDIDATES = [
    "/runpod-volume/weights/LongCat-Video",  # network volume, as mounted on serverless
    "/runpod-volume/LongCat-Video",
    "/workspace/weights/LongCat-Video",  # network volume, as mounted on a Pod
    "/weights/LongCat-Video",  # weights baked into the Docker image
    "./weights/LongCat-Video",
]

# Filled in by load_model() at cold start.
PIPE = None
LOAD_ERROR = None
LOAD_SECONDS = None
LOW_VRAM = False

# Loading 83 GB off a network volume takes minutes. It happens on a background thread so
# that the worker can answer a `{"health": true}` request straight away and tell you which
# stage it is in, instead of leaving your job sitting silently in the queue.
LOAD_STATE = {"stage": "starting", "started_at": time.time()}
LOADER_THREAD = None


# --------------------------------------------------------------------------------------
# Cold start: load the model once
# --------------------------------------------------------------------------------------


def torch_gc():
    """Give unused GPU memory back to the allocator."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def find_model_dir():
    """Return the folder that holds the LongCat-Video weights, or raise."""
    explicit = os.environ.get("MODEL_DIR")
    candidates = [explicit] if explicit else MODEL_DIR_CANDIDATES

    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "dit")):
            return path

    raise FileNotFoundError(
        "Could not find the LongCat-Video weights. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Attach your network volume to this endpoint and make sure the weights "
        "are at /runpod-volume/weights/LongCat-Video (see serverless/DEPLOY.md), "
        "or set the MODEL_DIR environment variable."
    )


def enable_text_encoder_offload(pipe):
    """
    Keep the 11 GB text encoder in CPU RAM and only move it to the GPU for the
    handful of milliseconds it is actually needed.

    The pipeline calls `self.encode_prompt(...)` internally, so replacing that one
    attribute on the instance is enough to wrap every text-encoding call.
    """
    original_encode_prompt = pipe.encode_prompt

    def encode_prompt_with_offload(*args, **kwargs):
        pipe.text_encoder.to("cuda")
        try:
            return original_encode_prompt(*args, **kwargs)
        finally:
            pipe.text_encoder.to("cpu")
            torch_gc()

    pipe.encode_prompt = encode_prompt_with_offload
    pipe.text_encoder.to("cpu")
    torch_gc()


def set_stage(stage):
    """Record and log which part of the (slow) start-up we are in."""
    LOAD_STATE["stage"] = stage
    elapsed = int(time.time() - LOAD_STATE["started_at"])
    print(f"[longcat] [{elapsed}s] {stage}", flush=True)


def load_model():
    """Load tokenizer / text encoder / VAE / DiT and both LoRAs onto the GPU."""
    global PIPE, LOAD_ERROR, LOAD_SECONDS, LOW_VRAM

    started = time.time()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU visible to this worker. A LongCat-Video endpoint must use a "
            "GPU worker (80 GB class recommended)."
        )

    set_stage("locating weights")
    checkpoint_dir = find_model_dir()
    print(f"[longcat] loading weights from {checkpoint_dir}", flush=True)

    # Single GPU: context-parallel size 1 -> split [1, 1]. Nothing distributed is
    # initialised, which is exactly what we want inside a serverless worker.
    cp_split_hw = context_parallel_util.get_optimal_split(1)

    set_stage("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    set_stage("loading text encoder (~23 GB)")
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    set_stage("loading vae")
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    set_stage("loading scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    set_stage("loading dit (~54 GB, the slow one)")
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
    )

    set_stage("moving model to gpu")
    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to("cuda")

    # The two LoRAs are parked in CPU RAM and only copied to the GPU while active
    # (see enable_loras / disable_all_loras in longcat_video/modules/longcat_video_dit.py),
    # so loading both here costs no permanent VRAM.
    cfg_step_lora = os.path.join(checkpoint_dir, "lora", "cfg_step_lora.safetensors")
    refinement_lora = os.path.join(checkpoint_dir, "lora", "refinement_lora.safetensors")
    if os.path.exists(cfg_step_lora):
        pipe.dit.load_lora(cfg_step_lora, "cfg_step_lora")
    if os.path.exists(refinement_lora):
        pipe.dit.load_lora(refinement_lora, "refinement_lora")

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    low_vram_env = os.environ.get("LOW_VRAM", "auto").lower()
    if low_vram_env in ("1", "true", "yes"):
        LOW_VRAM = True
    elif low_vram_env in ("0", "false", "no"):
        LOW_VRAM = False
    else:
        LOW_VRAM = total_vram_gb < 70

    if LOW_VRAM:
        print("[longcat] low-VRAM mode: text encoder will be offloaded to CPU", flush=True)
        enable_text_encoder_offload(pipe)

    PIPE = pipe
    LOAD_SECONDS = round(time.time() - started, 1)
    print(
        f"[longcat] model ready in {LOAD_SECONDS}s on "
        f"{torch.cuda.get_device_name(0)} ({total_vram_gb:.0f} GB)",
        flush=True,
    )


def loader_thread_target():
    global LOAD_ERROR
    try:
        load_model()
        LOAD_STATE["stage"] = "ready"
    except Exception as exc:  # noqa: BLE001 - reported back on the next request
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        LOAD_STATE["stage"] = "failed"
        print("[longcat] MODEL FAILED TO LOAD\n" + traceback.format_exc(), flush=True)


LOADER_THREAD = threading.Thread(target=loader_thread_target, daemon=True)
LOADER_THREAD.start()


def wait_until_loaded(timeout=None):
    """Block until the background load finishes. Returns an error string, or None."""
    LOADER_THREAD.join(timeout)
    if LOADER_THREAD.is_alive():
        return (
            f"The model is still loading (stage: {LOAD_STATE['stage']}) after "
            f"{int(time.time() - LOAD_STATE['started_at'])}s. Send "
            '{"input": {"health": true}} to watch its progress.'
        )
    return LOAD_ERROR


# --------------------------------------------------------------------------------------
# Request parsing
# --------------------------------------------------------------------------------------


def round_to(value, multiple):
    """Round to the nearest allowed multiple, never below one multiple."""
    return max(multiple, int(round(value / multiple)) * multiple)


def parse_request(job_input):
    """Turn the raw request JSON into validated, safe generation settings."""
    prompt = job_input.get("prompt")
    if not prompt or not str(prompt).strip():
        raise ValueError("'prompt' is required and cannot be empty.")

    mode = str(job_input.get("mode", "fast")).lower()
    if mode not in ("fast", "quality"):
        raise ValueError("'mode' must be either 'fast' or 'quality'.")

    width = round_to(int(job_input.get("width", 832)), SPATIAL_MULTIPLE)
    height = round_to(int(job_input.get("height", 480)), SPATIAL_MULTIPLE)
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"{width}x{height} is above the {MAX_PIXELS} pixel limit. Generate at "
            "832x480 and set 'refine': true to upscale to 720p instead."
        )

    # num_frames must be 4*k + 1 (93 frames @ 15 fps = 6.2 seconds).
    num_frames = int(job_input.get("num_frames", 93))
    num_frames = round_to(num_frames - 1, TEMPORAL_MULTIPLE) + 1
    num_frames = min(num_frames, MAX_FRAMES)

    # "fast" uses the distillation LoRA: 16 steps, no classifier-free guidance.
    # "quality" is the original 50-step setting with guidance.
    if mode == "fast":
        default_steps, default_guidance = 16, 1.0
    else:
        default_steps, default_guidance = 50, 4.0

    steps = min(int(job_input.get("num_inference_steps", default_steps)), MAX_STEPS)
    guidance = float(job_input.get("guidance_scale", default_guidance))

    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    if mode == "fast":
        negative_prompt = None  # the distilled model does not use a negative prompt

    return {
        "prompt": str(prompt).strip(),
        "negative_prompt": negative_prompt,
        "mode": mode,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": int(job_input.get("seed", 42)),
        "refine": bool(job_input.get("refine", False)),
        "spatial_refine_only": bool(job_input.get("spatial_refine_only", False)),
        "refine_steps": min(int(job_input.get("refine_steps", 50)), MAX_STEPS),
        "fps": int(job_input.get("fps", 15)),
        "crf": int(job_input.get("crf", 18)),
    }


# --------------------------------------------------------------------------------------
# Encoding / returning the video
# --------------------------------------------------------------------------------------


def frames_to_mp4(frames, path, fps, crf):
    """Write float frames in [0, 1] (N, H, W, 3) to an h264 mp4."""
    tensor = torch.from_numpy(np.asarray(frames))
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    write_video(path, tensor, fps=fps, video_codec="libx264", options={"crf": str(crf)})
    return tensor


def shrink_to_fit(tensor, path, fps, crf, max_bytes):
    """
    Re-encode with progressively stronger compression until the file fits in the
    response. Only used when no S3 bucket is configured.
    """
    while os.path.getsize(path) > max_bytes and crf < 34:
        crf += 6
        print(f"[longcat] video too big for an inline response, re-encoding at crf={crf}", flush=True)
        write_video(path, tensor, fps=fps, video_codec="libx264", options={"crf": str(crf)})
    return crf


def deliver(path, job_id, tensor, fps, crf):
    """Upload to S3 if configured, otherwise return base64 inside the JSON response."""
    max_bytes = int(float(os.environ.get("MAX_RESPONSE_MB", "7")) * 1024 * 1024)

    if os.environ.get("BUCKET_ENDPOINT_URL"):
        try:
            from runpod.serverless.utils import rp_upload

            url = rp_upload.upload_file_to_bucket(
                file_name=f"{job_id}.mp4", file_location=path
            )
            return {"video_url": url, "size_bytes": os.path.getsize(path)}
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the video
            print(f"[longcat] S3 upload failed ({exc}), returning base64 instead", flush=True)

    final_crf = shrink_to_fit(tensor, path, fps, crf, max_bytes)
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")

    result = {"video_base64": encoded, "size_bytes": size, "crf": final_crf}
    if size > max_bytes:
        result["warning"] = (
            "The video is still larger than the RunPod response limit even at the "
            "lowest quality. Use fewer frames, or configure BUCKET_ENDPOINT_URL to "
            "get an S3 link instead."
        )
    return result


# --------------------------------------------------------------------------------------
# The handler itself - RunPod calls this once per request
# --------------------------------------------------------------------------------------


def handler(job):
    job_input = job.get("input") or {}

    # Health check: answers immediately, even while the model is still loading, so you
    # can always tell the difference between "slow" and "broken".
    if job_input.get("health"):
        elapsed = int(time.time() - LOAD_STATE["started_at"])
        report = {
            "stage": LOAD_STATE["stage"],
            "seconds_since_worker_start": elapsed,
            "cuda_available": torch.cuda.is_available(),
        }
        if LOAD_ERROR is not None:
            report["status"] = "failed"
            report["error"] = LOAD_ERROR
            report["hint"] = "The full traceback is in the endpoint's Logs tab."
        elif LOADER_THREAD.is_alive():
            report["status"] = "loading"
            report["hint"] = (
                "Still reading weights off the network volume. Check again in a minute."
            )
        else:
            free, total = torch.cuda.mem_get_info()
            report["status"] = "ready"
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_total_gb"] = round(total / 1024**3, 1)
            report["vram_free_gb"] = round(free / 1024**3, 1)
            report["model_load_seconds"] = LOAD_SECONDS
            report["low_vram_mode"] = LOW_VRAM
        return report

    # A generation request has to wait for the model.
    load_error = wait_until_loaded()
    if load_error is not None:
        return {"error": load_error}

    try:
        cfg = parse_request(job_input)
    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}

    job_id = job.get("id", "output")
    timings = {}

    try:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(cfg["seed"])

        # ---- Stage 1: generate the base video ----------------------------------
        runpod.serverless.progress_update(job, "generating video")
        stage_started = time.time()

        use_distill = cfg["mode"] == "fast"
        if use_distill:
            PIPE.dit.enable_loras(["cfg_step_lora"])

        frames = PIPE.generate_t2v(
            prompt=cfg["prompt"],
            negative_prompt=cfg["negative_prompt"],
            height=cfg["height"],
            width=cfg["width"],
            num_frames=cfg["num_frames"],
            num_inference_steps=cfg["num_inference_steps"],
            use_distill=use_distill,
            guidance_scale=cfg["guidance_scale"],
            generator=generator,
        )[0]

        PIPE.dit.disable_all_loras()
        torch_gc()
        timings["generate_seconds"] = round(time.time() - stage_started, 1)

        fps = cfg["fps"]

        # ---- Stage 2 (optional): refine to 720p --------------------------------
        if cfg["refine"]:
            runpod.serverless.progress_update(job, "refining to 720p")
            stage_started = time.time()

            stage1_video = [
                PIL.Image.fromarray((frames[i] * 255).astype(np.uint8))
                for i in range(frames.shape[0])
            ]
            del frames
            torch_gc()

            PIPE.dit.enable_loras(["refinement_lora"])
            PIPE.dit.enable_bsa()
            try:
                frames = PIPE.generate_refine(
                    prompt=cfg["prompt"],
                    stage1_video=stage1_video,
                    num_inference_steps=cfg["refine_steps"],
                    generator=generator,
                    spatial_refine_only=cfg["spatial_refine_only"],
                )[0]
            finally:
                PIPE.dit.disable_all_loras()
                PIPE.dit.disable_bsa()
                torch_gc()

            # Refinement also doubles the frame rate unless it is spatial-only.
            if not cfg["spatial_refine_only"]:
                fps = fps * 2
            timings["refine_seconds"] = round(time.time() - stage_started, 1)

        # ---- Encode and return --------------------------------------------------
        runpod.serverless.progress_update(job, "encoding mp4")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, f"{job_id}.mp4")
            tensor = frames_to_mp4(frames, path, fps, cfg["crf"])
            del frames
            payload = deliver(path, job_id, tensor, fps, cfg["crf"])

        payload.update(
            {
                "seed": cfg["seed"],
                "mode": cfg["mode"],
                "width": cfg["width"],
                "height": cfg["height"],
                "num_frames": cfg["num_frames"],
                "fps": fps,
                "refined": cfg["refine"],
                "timings": timings,
            }
        )
        return payload

    except torch.cuda.OutOfMemoryError:
        torch_gc()
        return {
            "error": (
                "The GPU ran out of memory. Try a smaller resolution or fewer frames, "
                "turn off 'refine', or move the endpoint to an 80 GB GPU."
            )
        }
    except Exception as exc:  # noqa: BLE001 - surface the real reason to the caller
        torch_gc()
        print(traceback.format_exc(), flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        torch_gc()


# `concurrency_modifier` returning 1 keeps a worker on a single job at a time - video
# generation already saturates the GPU, and two jobs at once would just run out of memory.
runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda _: 1})
