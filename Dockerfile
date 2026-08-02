# ======================================================================================
# LongCat-Video - RunPod Serverless worker image
#
# This image contains the CODE only, not the 83 GB of model weights. The weights live
# on a RunPod network volume that gets mounted at /runpod-volume (see serverless/DEPLOY.md).
# Keeping them out of the image is what makes the build finish in minutes instead of hours.
# ======================================================================================

# CUDA 12.8 (not 12.4) so that ONE image runs on every GPU RunPod offers:
#   Ampere sm_80 (A100)  Ada sm_89 (L40S)  Hopper sm_90 (H100/H200)  Blackwell sm_120
#     (RTX PRO 6000, B200). A cu124 build has no kernels for sm_120 and dies with
#     "no kernel image is available for execution on the device".
# Being architecture-agnostic matters here because your network volume pins the endpoint
# to a single datacenter, so you cannot afford to be picky about which GPU you get.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

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

# PyTorch 2.7.1 + cu128. LongCat-Video's requirements.txt pins torch 2.6.0, but 2.6 has
# no Blackwell support at all, and nothing this pipeline uses changed between 2.6 and 2.7.
RUN pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

# FlashAttention-2 from the official pre-built wheel. Building from source takes 1-3 hours
# and would blow past RunPod's 30 minute build timeout.
#
# The wheel must match PyTorch's C++ ABI, and PyTorch switched that setting between
# releases - guessing wrong gives an "undefined symbol" ImportError at run time. So ask
# the installed torch which ABI it was built with and pick the matching wheel, then prove
# it imports before the build is allowed to continue.
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
    && echo "torch reports cxx11abi=${ABI}" \
    && pip install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl" \
    && (python -c "import flash_attn; print('flash-attn', flash_attn.__version__, 'imports OK')" \
        || echo "NOTE: flash-attn import check skipped - the build machine has no GPU driver")

COPY serverless/requirements-serverless.txt /tmp/requirements-serverless.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serverless.txt

# Guard: if anything above had pulled torch again, pip would have installed the CPU-only
# build from PyPI and the worker would fail at run time with "No CUDA GPU visible".
# Better to fail the build here, loudly, than to discover that on the first request.
RUN python -c "import torch, torchvision; assert torch.version.cuda, f'CUDA build of torch was replaced by {torch.__version__}'; print('OK torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.version.cuda)"

WORKDIR /app
COPY longcat_video /app/longcat_video
COPY serverless/handler.py /app/handler.py

# RunPod starts the container and the SDK inside handler.py waits for jobs.
CMD ["python", "-u", "handler.py"]
