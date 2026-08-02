#!/usr/bin/env bash
# ======================================================================================
# ONE-TIME SETUP - run this INSIDE a temporary RunPod Pod, not on your Mac.
#
# It downloads the LongCat-Video weights (~83 GB) onto your network volume, then
# converts the two biggest folders to bfloat16 (~83 GB -> ~44 GB). The conversion is
# lossless for us, because the model is loaded as bfloat16 at run time anyway - it just
# halves the amount of data every cold start has to read.
#
# Usage inside the Pod's web terminal:
#     bash setup_weights.sh
# ======================================================================================

set -euo pipefail

# On a Pod, the network volume is mounted at /workspace.
# (The same volume appears at /runpod-volume on serverless workers.)
VOLUME_DIR="${VOLUME_DIR:-/workspace}"
TARGET_DIR="$VOLUME_DIR/weights/LongCat-Video"

echo "==> Installing the Hugging Face downloader"
pip install --quiet --upgrade "huggingface_hub" hf_transfer safetensors torch --no-input

echo "==> Downloading meituan-longcat/LongCat-Video into $TARGET_DIR"
echo "    This is ~83 GB and usually takes 15-40 minutes."
mkdir -p "$TARGET_DIR"

HF_HUB_ENABLE_HF_TRANSFER=1 python - "$TARGET_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

target = sys.argv[1]
snapshot_download(
    repo_id="meituan-longcat/LongCat-Video",
    local_dir=target,
    max_workers=8,
    # Re-running this script resumes instead of starting over.
)
print("download complete ->", target)
PY

echo "==> Shrinking the weights to bfloat16 (halves every future cold start)"
python "$(dirname "$0")/convert_to_bf16.py" "$TARGET_DIR"

echo
echo "==> Done. Weights are at: $TARGET_DIR"
du -sh "$TARGET_DIR"
echo
echo "On your serverless endpoint the same files will be visible at:"
echo "    /runpod-volume/weights/LongCat-Video"
echo "You can now STOP AND DELETE this Pod - the network volume keeps the files."
