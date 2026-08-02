#!/usr/bin/env python3
"""
Send a prompt to your RunPod Serverless endpoint and save the resulting mp4.

Run this on YOUR MAC. It uses only the Python standard library, so there is nothing
to install - the `python3` that ships with macOS is enough.

    export RUNPOD_API_KEY=rpa_xxxxxxxxxxxxxxxx
    export RUNPOD_ENDPOINT_ID=abc123xyz
    python3 serverless/client.py "a red fox running through deep snow at sunrise"

Useful options:
    --quality            50 diffusion steps instead of 16 (slower, a bit better)
    --refine             add the 720p refinement pass (roughly doubles the cost)
    --frames 121         longer clip; must be 4*k+1, e.g. 61, 93, 121
    --seed 7             change the seed to get a different video from the same prompt
    --health             just check that the endpoint is alive
    --job-id <id>        reconnect to a job already running (Ctrl+C does not cancel it)

Be patient on the first call after a build: RunPod has to pull a 12 GB image and load
83 GB of weights before any generation starts. Five to ten minutes is normal.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE_URL = "https://api.runpod.ai/v2"


def post_json(url, payload, api_key):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url, api_key):
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}, method="GET"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def build_input(args):
    if args.health:
        return {"health": True}

    job_input = {
        "prompt": args.prompt,
        "mode": "quality" if args.quality else "fast",
        "width": args.width,
        "height": args.height,
        "num_frames": args.frames,
        "seed": args.seed,
        "refine": args.refine,
    }
    if args.steps:
        job_input["num_inference_steps"] = args.steps
    if args.negative_prompt:
        job_input["negative_prompt"] = args.negative_prompt
    return job_input


STATE_LABELS = {
    "IN_QUEUE": "queued - no worker has picked this up yet",
    "IN_PROGRESS": "a worker is running this job",
}


def show_endpoint_status(endpoint, api_key):
    """
    Print RunPod's own view of the endpoint: how many workers exist, what state they
    are in, and how many jobs are waiting. This is the fastest way to tell whether a
    long wait is normal slowness or an actual fault.
    """
    health = get_json(f"{endpoint}/health", api_key)
    workers = health.get("workers", {})
    jobs = health.get("jobs", {})

    print("workers:", json.dumps(workers))
    print("jobs:   ", json.dumps(jobs))
    print()

    if workers.get("unhealthy"):
        print("DIAGNOSIS: a worker is UNHEALTHY - the container is crashing on start-up.")
        print("  Open the endpoint's Logs tab and look for a Python traceback.")
    elif workers.get("throttled"):
        print("DIAGNOSIS: workers are THROTTLED - RunPod has no free GPU of your chosen")
        print("  type in your network volume's datacenter. Allow more GPU types.")
    elif workers.get("initializing") and not workers.get("ready") and not workers.get("running"):
        print("DIAGNOSIS: the worker is still INITIALIZING - pulling the 12 GB image.")
        print("  Normal for the first few minutes after a build. If it lasts more than")
        print("  ~15 minutes, check the Logs tab.")
    elif jobs.get("inQueue") and not workers.get("running"):
        print("DIAGNOSIS: jobs are queued but no worker is running them.")
        print("  Check that Max Workers is at least 1 and look at the Logs tab.")
    elif workers.get("running"):
        print("DIAGNOSIS: a worker is running your job. This is the healthy case -")
        print("  loading 83 GB of weights genuinely takes several minutes.")
    elif workers.get("idle") or workers.get("ready"):
        print("DIAGNOSIS: a worker is warm and waiting. Requests should be fast now.")
    else:
        print("DIAGNOSIS: no workers at all. If jobs are queued, check Max Workers > 0.")


def wait_for_job(endpoint, job_id, api_key, poll_seconds=5, quiet_tick=30):
    """
    Poll until the job reaches a final state. Returns the final status dict.

    Prints on every state change and on progress messages from the worker, plus a
    heartbeat every 30 seconds so a long wait does not look like a hang.
    """
    started = time.time()
    last_note = None
    last_state = None
    last_tick = 0

    while True:
        time.sleep(poll_seconds)
        elapsed = int(time.time() - started)
        status = get_json(f"{endpoint}/status/{job_id}", api_key)
        state = status.get("status")

        # While the job runs, `output` holds the worker's latest progress string.
        note = status.get("output") if isinstance(status.get("output"), str) else None
        if note and note != last_note:
            print(f"  [{elapsed}s] {note}")
            last_note = note
        elif state != last_state:
            print(f"  [{elapsed}s] {STATE_LABELS.get(state, state)}")
        elif elapsed - last_tick >= quiet_tick:
            # Always name the actual state - "still working" hid whether the job was
            # queued (nothing has picked it up) or genuinely running.
            print(f"  [{elapsed}s] still {STATE_LABELS.get(state, state)}")
            last_tick = elapsed

        if state != last_state:
            last_state = state
            last_tick = elapsed

        if state == "COMPLETED":
            return status
        if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.exit(f"Job {state}:\n{json.dumps(status, indent=2)}")


def save_output(output, out_path):
    """Write the mp4 from either the inline base64 or the S3 link."""
    if output.get("video_base64"):
        with open(out_path, "wb") as handle:
            handle.write(base64.b64decode(output["video_base64"]))
        return out_path

    if output.get("video_url"):
        print(f"downloading {output['video_url']}")
        with urllib.request.urlopen(output["video_url"], timeout=300) as response:
            with open(out_path, "wb") as handle:
                handle.write(response.read())
        return out_path

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate a video on RunPod Serverless.")
    parser.add_argument("prompt", nargs="?", default="", help="what the video should show")
    parser.add_argument("--quality", action="store_true", help="50 steps with guidance")
    parser.add_argument("--refine", action="store_true", help="add the 720p refinement pass")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=93, help="must be 4*k+1")
    parser.add_argument("--steps", type=int, default=None, help="override diffusion steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", dest="negative_prompt", default=None)
    parser.add_argument("--out", default=None, help="output file name")
    parser.add_argument("--health", action="store_true", help="only ping the endpoint")
    parser.add_argument(
        "--status",
        action="store_true",
        help="show worker/queue state and diagnose a stuck endpoint (no job submitted)",
    )
    parser.add_argument(
        "--job-id",
        dest="job_id",
        default=None,
        help="reconnect to a job already running instead of submitting a new one",
    )
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        sys.exit(
            "Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID first, for example:\n"
            "  export RUNPOD_API_KEY=rpa_xxxxxxxx\n"
            "  export RUNPOD_ENDPOINT_ID=abc123xyz"
        )
    if not args.prompt and not args.health and not args.job_id and not args.status:
        sys.exit('Give a prompt, e.g.  python3 serverless/client.py "a cat on a skateboard"')

    endpoint = f"{BASE_URL}/{endpoint_id}"
    started = time.time()

    if args.status:
        show_endpoint_status(endpoint, api_key)
        return

    if args.job_id:
        # Reconnect to a job that is already running (e.g. after pressing Ctrl+C).
        job_id = args.job_id
        print(f"reconnecting to job {job_id} ...")
    else:
        try:
            submitted = post_json(f"{endpoint}/run", {"input": build_input(args)}, api_key)
        except urllib.error.HTTPError as error:
            sys.exit(
                f"RunPod rejected the request ({error.code}): {error.read().decode('utf-8')}"
            )

        job_id = submitted.get("id")
        if not job_id:
            sys.exit(f"Unexpected response from RunPod: {submitted}")

        print(f"job {job_id} submitted, waiting for the worker ...")
        print(
            "The FIRST request after a build is slow: RunPod pulls a 12 GB image, then\n"
            "the worker loads 83 GB of weights. Five to ten minutes is normal. Later\n"
            "requests take ~2 minutes cold, or seconds while the worker is still warm.\n"
        )

    try:
        status = wait_for_job(endpoint, job_id, api_key)
    except KeyboardInterrupt:
        sys.exit(
            "\n\nStopped watching - but the job is STILL RUNNING on RunPod.\n"
            "Pressing Ctrl+C here does not cancel it. Reconnect with:\n\n"
            f"    python3 serverless/client.py --job-id {job_id}\n"
        )

    output = status.get("output") or {}
    if output.get("error"):
        sys.exit(f"The worker returned an error: {output['error']}")

    if args.health:
        print(json.dumps(output, indent=2))
        return

    if not (output.get("video_base64") or output.get("video_url")):
        # Happens when you reconnect with --job-id to what was actually a health check.
        print(json.dumps(output, indent=2))
        return

    out_path = args.out or f"longcat_{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}.mp4"
    saved = save_output(output, out_path)
    if not saved:
        sys.exit(f"No video in the response: {json.dumps(output, indent=2)}")

    elapsed = int(time.time() - started)
    size_mb = os.path.getsize(saved) / 1024**2
    print(f"\nsaved {saved} ({size_mb:.1f} MB) in {elapsed}s")
    if output.get("timings"):
        print(f"worker timings: {output['timings']}")
    if output.get("warning"):
        print(f"warning: {output['warning']}")


if __name__ == "__main__":
    main()
