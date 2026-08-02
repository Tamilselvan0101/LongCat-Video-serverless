# Deploying LongCat-Video on RunPod Serverless

A complete, copy-paste guide. It assumes you have never written code. Every command
below is meant to be pasted exactly as written into a terminal.

You will end up with a private API you can call from your Mac:

```
python3 serverless/client.py "a red fox running through deep snow at sunrise"
  -> longcat_20260801_143022_seed42.mp4
```

You only pay while a video is actually being generated. When you are not using it,
the cost is a few dollars a month for storing the model weights.

---

## 0. What was added to your project, and why

| File                                       | What it is                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| `serverless/handler.py`                  | The server. Loads the model once, then turns each request into an mp4.         |
| `Dockerfile`                             | The recipe RunPod uses to build your worker (Python, PyTorch, FlashAttention). |
| `serverless/requirements-serverless.txt` | The exact Python package versions the worker needs.                            |
| `serverless/setup_weights.sh`            | One-time script that downloads the 83 GB model onto your storage.              |
| `serverless/convert_to_bf16.py`          | Shrinks the model 83 GB → ~44 GB so cold starts are twice as fast.            |
| `serverless/client.py`                   | Runs on your Mac. Sends a prompt, saves the mp4.                               |
| `serverless/test_input.json`             | A sample request, used for testing.                                            |
| `.dockerignore`                          | Keeps junk out of the build.                                                   |

**The original `run_demo_text_to_video.py` cannot be used as-is on serverless.** It
requires `torchrun` and calls `torch.distributed.init_process_group`, which is for
multi-GPU clusters — on a single serverless worker it hangs forever. It also hard-codes
one prompt and writes files to a disk that disappears when the worker shuts down.
`handler.py` is the same model, wired for one GPU, one request at a time, results
returned over HTTP. Your original files are untouched and still work locally.

---

## 1. What it costs (estimates — check the RunPod console for current rates)

| Item                  | Price                                                                                   | Notes                                                                 |
| --------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Network storage       | ~$0.07 / GB / month | 120 GB ≈**$8.40/month**, charged whether you use it or not |                                                                       |
| A100 80 GB serverless | ~$2.72 / hour | ≈ $0.045 per minute of generation                                      |                                                                       |
| H100 80 GB serverless | ~$4.55 / hour | ≈ $0.076 per minute, roughly 2× faster than A100                      |                                                                       |
| L40S 48 GB serverless | ~$1.75 / hour                                                                           | Cheapest, but 480p only — not enough memory for the 720p refine pass |

Rough per-video cost on an A100 80 GB:

* 6-second 480p clip, `fast` mode, worker already warm: **~$0.10**
* Same clip but the worker was asleep (adds ~2 min of model loading): **~$0.20**
* With `--refine` (720p, 30 fps): **~$0.40 – $0.70**

The handler returns a `timings` field with the real numbers so you can check this
against your own usage after the first few videos.

---

## 2. Before you start

You need:

1. A **RunPod account** with some credit loaded — [https://runpod.io](https://runpod.io)
2. A **GitHub account** — [https://github.com](https://github.com) (RunPod will build your worker straight
   from a GitHub repo, so you never have to install Docker on your Mac)
3. About **an hour**, most of it waiting for downloads

---

## Part 1 — Put your code on GitHub

RunPod needs to read your code from GitHub. Open the **Terminal** app on your Mac
(press `Cmd+Space`, type "Terminal", press Enter) and paste these commands one at a time.

**1.1 — Go to your project folder:**

```bash
cd ~/Downloads/LongCat-Video-main
```

**1.2 — Turn it into a git repository:**

```bash
git init
git add .
git commit -m "LongCat-Video with RunPod serverless handler"
```

If macOS asks you to install "command line developer tools", click **Install**, wait for
it to finish, then run the commands again.

If git complains that it doesn't know who you are, run these two lines first (use your
own email and name), then repeat the commit:

```bash
git config --global user.email "tamilselvanking0@gmail.com"
git config --global user.name "Tamilselvan"
```

**1.3 — Create an empty repository on GitHub:**

Go to [https://github.com/new](https://github.com/new). Name it `LongCat-Video-serverless`. Choose **Private**.
Do **not** tick "Add a README". Click **Create repository**.

**1.4 — Push your code up.** Replace `YOUR-USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR-USERNAME/LongCat-Video-serverless.git
git branch -M main
git push -u origin main
```

GitHub will ask for a username and password. The "password" must be a **personal access
token**, not your normal password: go to [https://github.com/settings/tokens](https://github.com/settings/tokens), click
*Generate new token (classic)*, tick the `repo` checkbox, generate it, and paste the
`ghp_...` string as the password.

> The model weights are **not** in this upload — the `weights/` folder is ignored on
> purpose. GitHub cannot hold 83 GB. The weights go on RunPod storage in Part 2.

---

## Part 2 — Put the model weights on RunPod storage

The model is 83 GB. It cannot go inside the worker image, and re-downloading it on every
request would be absurdly slow. Instead it lives on a **network volume**: a permanent
disk that your serverless workers mount instantly at startup.

**2.1 — Create the network volume**

1. In the RunPod console, go to **Storage** → **New Network Volume**.
2. **Datacenter**: pick one that has plenty of GPUs, e.g. `EU-RO-1` or `US-KS-2`.
   ⚠️ **Write this datacenter down.** Your serverless endpoint must run in the same
   datacenter, so this choice limits which GPUs you can use later.
3. **Size**: `120` GB. (100 GB is the bare minimum; you can grow a volume later but you
   can never shrink it, so 120 GB avoids a painful redo for about $1.40/month extra.)
4. **Name**: `longcat-weights`. Click **Create**.

**2.2 — Start a temporary Pod to fill the volume**

A serverless worker cannot download 83 GB for you, so you rent a cheap machine for an
hour to do it once.

1. Go to **Pods** → **Deploy**.
2. On the left, under **Network Volume**, select `longcat-weights`. The datacenter is
   now locked to match.
3. Pick the **cheapest GPU available** — you are only downloading files, the GPU is
   irrelevant. An RTX A4000 or similar is fine. Make sure it has at least **32 GB of
   system RAM** (shown in the card) for the conversion step.
4. Template: **RunPod PyTorch** (any recent version).
5. Click **Deploy On-Demand**. Wait until the pod shows **Running**.

**2.3 — Download the weights**

Click **Connect** → **Start Web Terminal** → **Connect to Web Terminal**. A black
terminal opens in your browser. Paste this, replacing `YOUR-USERNAME`:

```bash
cd /workspace
git clone https://github.com/YOUR-USERNAME/LongCat-Video-serverless.git code
bash code/serverless/setup_weights.sh
```

(For a private repo, git will ask for your username and the `ghp_...` token again.)

This will:

* download all 83 GB into `/workspace/weights/LongCat-Video` (15–40 minutes), then
* convert the big files from float32 to bfloat16, ending at about 44 GB (10–20 minutes).

You will see the final size printed at the end. **If your internet connection to the web
terminal drops, just run `bash code/serverless/setup_weights.sh` again** — it resumes
where it left off and skips files that are already converted.

**2.4 — Delete the Pod**

Back in the **Pods** list, click **Terminate** on the pod. This is important — a running
pod bills by the hour. **Your files are safe**: they are on the network volume, not on
the pod.

---

## Part 3 — Create the Serverless endpoint

**3.1 — Connect GitHub to RunPod**

In the RunPod console: **Settings** → **Connections** → **GitHub** → **Connect**.
Authorize RunPod and grant it access to your `LongCat-Video-serverless` repository.

**3.2 — Create the endpoint**

1. Go to **Serverless** → **New Endpoint**.
2. Choose **Import Git Repository** and select `LongCat-Video-serverless`.
3. **Branch**: `main`. **Dockerfile path**: `Dockerfile` (it is in the repo root).
4. **Endpoint name**: `longcat-video`.
5. **Endpoint Type**: `Queue`.

**3.3 — GPU and worker settings**

| Setting           | Value                              | Why                                                                                                                                                                                                        |
| ----------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU               | **80 GB** (H100 / A100 80GB) | The model needs ~40 GB just for weights; 80 GB leaves room for the 720p refine pass. Pick 48 GB only if you will stay at 480p and never use`refine`.                                                     |
| Active Workers    | **0**                        | ⚠️ Anything above 0 bills you 24/7 even when idle. Keep it at 0.                                                                                                                                         |
| Max Workers       | **1**                        | You are the only user. This also caps your worst-case spend.                                                                                                                                               |
| Idle Timeout      | **10** seconds               | How long a worker stays warm after finishing. Raise to 120 s if you plan to generate several videos back to back — you skip the 2-minute model load each time, at the price of paying for that idle time. |
| Execution Timeout | **1800** seconds (30 min)    | Long enough for a refine pass; kills anything stuck.                                                                                                                                                       |
| FlashBoot         | **Enabled**                  | Free, makes repeat cold starts faster.                                                                                                                                                                     |
| Container Disk    | **25** GB                    | The image is ~12 GB, plus working space.                                                                                                                                                                   |

**3.4 — Attach the network volume**

Still on the creation page, under **Storage** / **Network Volume**, select
`longcat-weights`. **This is the step people forget** — without it the worker starts,
finds no model, and every request fails with "Could not find the LongCat-Video weights".

Note that the GPU list will now only show GPUs available in that volume's datacenter.

**3.5 — Environment variables (optional)**

None are required. The handler auto-detects everything. You may add:

| Name                | Value                                    | Effect                                                         |
| ------------------- | ---------------------------------------- | -------------------------------------------------------------- |
| `MODEL_DIR`       | `/runpod-volume/weights/LongCat-Video` | Only if you put the weights somewhere else                     |
| `LOW_VRAM`        | `1`                                    | Force the memory-saving mode on a 48 GB GPU                    |
| `MAX_RESPONSE_MB` | `7`                                    | Largest video returned inline before it gets compressed harder |

**3.6 — Deploy**

Click **Create Endpoint**. Open the **Builds** tab and watch the Docker build — it takes
**10–20 minutes** the first time. When it says *Build Successful* and the endpoint shows
workers as `Idle`, you are ready.

Copy your **Endpoint ID** (a short string like `abc123xyz`) from the top of the page.

---

## Part 4 — Generate your first video

**4.1 — Get an API key**: RunPod console → **Settings** → **API Keys** → **Create API
Key**. Copy the `rpa_...` value; it is shown only once.

**4.2 — In Terminal on your Mac**, paste these (with your own two values):

```bash
cd ~/Downloads/LongCat-Video-main
export RUNPOD_API_KEY=rpa_xxxxxxxxxxxxxxxxxxxx
export RUNPOD_ENDPOINT_ID=abc123xyz
```

**4.3 — Check the endpoint is healthy** (this costs a few cents and confirms the model
loaded from your volume):

```bash
python3 serverless/client.py --health
```

You should see something like:

```json
{
  "status": "ready",
  "gpu": "NVIDIA A100 80GB PCIe",
  "vram_total_gb": 79.2,
  "model_load_seconds": 96.4,
  "low_vram_mode": false
}
```

**4.4 — Make a video:**

```bash
python3 serverless/client.py "a red fox running through deep snow at sunrise, cinematic, golden light"
```

The first run takes ~3–4 minutes (the worker has to wake up and load the model). The mp4
lands in your current folder. Open it with `open longcat_*.mp4`.

**4.5 — Other things you can ask for:**

```bash
# higher quality: 50 diffusion steps with guidance instead of the fast 16-step mode
python3 serverless/client.py "a lighthouse in a storm" --quality

# upscale to 720p / 30 fps afterwards (slower and more expensive, much sharper)
python3 serverless/client.py "a lighthouse in a storm" --refine

# a longer clip - the number must be 4×k+1, so 61, 93, 121, 181 ...
python3 serverless/client.py "timelapse of clouds over a desert" --frames 121

# same prompt, different result
python3 serverless/client.py "a lighthouse in a storm" --seed 7
```

Every option is listed by `python3 serverless/client.py --help`.

---

## Part 5 — Keeping the bill small

* **Active Workers must stay at 0.** This is the single biggest way to accidentally spend
  money on RunPod. At 0, an idle endpoint costs nothing.
* **Max Workers = 1** means you can never accidentally run several $4/hour GPUs at once.
* **Idle Timeout** is the knob for "fast repeat requests" vs "cheap". 10 seconds = you pay
  almost nothing between videos but reload the model each time. 300 seconds = instant
  follow-up videos for 5 minutes of GPU billing.
* **`fast` mode is the default** for a reason: 16 diffusion steps instead of 50, and it
  looks very close to `--quality` for most prompts. Only reach for `--quality` or
  `--refine` on a prompt you already like.
* The **network volume bills continuously** (~$8.40/month for 120 GB) whether or not you
  generate anything. If you stop using the project for a long time, delete the volume —
  but you will have to redo Part 2 to come back.

---

## Part 6 — Changing the code later

RunPod rebuilds your worker from **GitHub releases**, not from every push. After editing
a file:

```bash
cd ~/Downloads/LongCat-Video-main
git add .
git commit -m "describe what you changed"
git push
```

Then in the RunPod console open your endpoint → **Builds** → **New Build**, or create a
new release on GitHub. Watch the Builds tab until it succeeds.

---

## Troubleshooting

**"Could not find the LongCat-Video weights"**
The network volume is not attached to the endpoint, or the files are in the wrong place.
Edit the endpoint and attach `longcat-weights`. To verify the contents, start a temporary
pod with that volume and run `ls /workspace/weights/LongCat-Video` — you should see
`dit`, `text_encoder`, `vae`, `tokenizer`, `scheduler`, `lora`.

**The job sits in `IN_QUEUE` for a long time**
Normal on the first request: RunPod is pulling a 12 GB image and loading 44 GB of weights.
Expect 3–5 minutes cold, ~1–2 minutes on later cold starts thanks to FlashBoot. If it
lasts more than 10 minutes, check the **Workers** tab — "no GPUs available" means the
datacenter your volume lives in has none of your selected GPU type free right now; edit
the endpoint and allow more GPU types.

**"The GPU ran out of memory"**
You are on a 48 GB GPU, or asking for too much. Drop `--refine`, lower `--frames`, or move
the endpoint to an 80 GB GPU. Setting the `LOW_VRAM=1` environment variable frees about
11 GB by keeping the text encoder in system RAM.

**Build fails in the Builds tab**
Read the log from the bottom. A network hiccup while downloading PyTorch or FlashAttention
is the usual cause — click **New Build** and try again. RunPod caps builds at 30 minutes.

**The video looks over-compressed**
Without an S3 bucket the handler must squeeze the mp4 under RunPod's ~10 MB response
limit, so long or refined videos get re-encoded at lower quality. Two fixes: ask for fewer
frames, or set `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID` and `BUCKET_SECRET_ACCESS_KEY`
in the endpoint's environment variables — the handler then uploads the full-quality file to
your own S3/Cloudflare R2 bucket and returns a download link instead.

**Job returns `FAILED` with no useful message**
Open the endpoint → **Logs**. The Python traceback from the worker is there in full.

---

## Reference: the request format

If you ever want to call the endpoint from something other than `client.py` — a script,
Shortcuts, n8n, whatever — this is what it accepts. Send it to
`https://api.runpod.ai/v2/<ENDPOINT_ID>/run` with the header
`Authorization: Bearer <API_KEY>`.

```jsonc
{
  "input": {
    "prompt": "required - what the video should show",
    "negative_prompt": "optional - what to avoid (ignored in fast mode)",
    "mode": "fast",            // "fast" = 16 steps (default), "quality" = 50 steps
    "width": 832,              // multiple of 16
    "height": 480,             // multiple of 16
    "num_frames": 93,          // must be 4*k+1; 93 frames @ 15fps = 6.2 seconds
    "num_inference_steps": 16, // optional override
    "guidance_scale": 1.0,     // optional override
    "seed": 42,
    "refine": false,           // true = second pass, 720p at 30fps
    "spatial_refine_only": false, // true = upscale but keep 15fps
    "fps": 15,
    "crf": 18                  // mp4 quality, lower = bigger file
  }
}
```

The response looks like:

```jsonc
{
  "video_base64": "AAAAIGZ0eXBpc29t...",  // or "video_url" if you configured S3
  "size_bytes": 3120544,
  "seed": 42,
  "mode": "fast",
  "width": 832, "height": 480, "num_frames": 93, "fps": 15,
  "refined": false,
  "timings": { "generate_seconds": 71.3 }
}
```

Send `{"input": {"health": true}}` for a free-ish status check instead of a generation.
