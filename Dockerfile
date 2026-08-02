# ======================================================================================
# LongCat-Video - RunPod Serverless worker image
#
# This image contains the CODE only, not the 83 GB of model weights. The weights live
# on a RunPod network volume that gets mounted at /runpod-volume (see serverless/DEPLOY.md).
# Keeping them out of the image is what makes the build finish in minutes instead of hours.
# ======================================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Ubuntu 22.04 ships Python 3.10, which is the version LongCat-Video targets.
# ffmpeg / libgl1 / libglib2.0-0 are needed by the video and image libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

# PyTorch 2.6.0 built for CUDA 12.4, matching the version LongCat-Video was tested on.
RUN pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# FlashAttention-2 from the official pre-built wheel. Building it from source needs a
# GPU-less compile that takes 1-3 hours and would blow past RunPod's 30 minute build
# timeout, so we always install the binary that matches cp310 + torch 2.6 + cu12.
RUN pip install --no-cache-dir \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY serverless/requirements-serverless.txt /tmp/requirements-serverless.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serverless.txt

WORKDIR /app
COPY longcat_video /app/longcat_video
COPY serverless/handler.py /app/handler.py

# RunPod starts the container and the SDK inside handler.py waits for jobs.
CMD ["python", "-u", "handler.py"]
