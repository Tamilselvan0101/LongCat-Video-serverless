#!/usr/bin/env bash
# ======================================================================================
# ONE-TIME SETUP - run this INSIDE a temporary RunPod Pod, not on your Mac.
#
# By default it ONLY downloads the LongCat-Video weights (~83 GB, untouched) onto your
# network volume.
#
#     bash setup_weights.sh
#
# OPTIONAL: add --convert-bf16 to also shrink the checkpoint from float32 to bfloat16
# (~83 GB -> ~44 GB), which roughly halves how long each cold start takes to read the
# model. It does NOT change your videos - handler.py loads the model as bfloat16 either
# way, so the GPU ends up with identical weights. But it is IRREVERSIBLE: you would have
# to re-download to get float32 back.
#
#     bash setup_weights.sh --convert-bf16
# ======================================================================================

set -euo pipefail

CONVERT_BF16=0
for arg in "$@"; do
    case "$arg" in
        --convert-bf16) CONVERT_BF16=1 ;;
        *) echo "unknown option: $arg"; echo "usage: bash setup_weights.sh [--convert-bf16]"; exit 1 ;;
    esac
done

# On a Pod, the network volume is mounted at /workspace.
# (The same volume appears at /runpod-volume on serverless workers.)
VOLUME_DIR="${VOLUME_DIR:-/workspace}"
TARGET_DIR="$VOLUME_DIR/weights/LongCat-Video"

# Say up front exactly what is about to happen, so a stale copy of this script can
# never quietly modify your weights.
echo "======================================================================"
echo " LongCat-Video weight setup"
echo " target:              $TARGET_DIR"
if [ "$CONVERT_BF16" = "1" ]; then
    echo " bfloat16 conversion: ENABLED  (float32 originals will be REPLACED)"
    echo
    echo " Press Ctrl+C within 10 seconds to abort."
    sleep 10
else
    echo " bfloat16 conversion: DISABLED (original float32 weights are kept)"
fi
echo "======================================================================"
echo

echo "==> Installing the Hugging Face downloader"
# Note: no --upgrade for torch. The pod template already has a working PyTorch and
# replacing it would download several more GB for no reason.
pip install --quiet --upgrade huggingface_hub hf_transfer --no-input
if [ "$CONVERT_BF16" = "1" ]; then
    pip install --quiet safetensors --no-input
fi

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

if [ "$CONVERT_BF16" = "1" ]; then
    echo "==> Shrinking the weights to bfloat16 (halves every future cold start)"
    python "$(dirname "$0")/convert_to_bf16.py" "$TARGET_DIR"
else
    echo "==> Keeping the original float32 weights (no conversion)."
    echo "    Re-run with --convert-bf16 later if you want faster cold starts."
fi

echo
echo "==> Done. Weights are at: $TARGET_DIR"
du -sh "$TARGET_DIR"
echo
echo "On your serverless endpoint the same files will be visible at:"
echo "    /runpod-volume/weights/LongCat-Video"
echo "You can now STOP AND DELETE this Pod - the network volume keeps the files."
