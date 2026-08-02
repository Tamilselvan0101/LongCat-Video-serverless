"""
OPTIONAL. Shrink the LongCat-Video checkpoint from float32 to bfloat16, in place.

You do not need to run this. Everything works with the original float32 weights; this
only makes cold starts faster, and it cannot be undone without re-downloading 83 GB.

Why it does not change your videos: the published checkpoint stores the DiT and the text
encoder in float32 (~77 GB),
but `handler.py` loads them with `torch_dtype=torch.bfloat16`, so every byte of that
extra precision is thrown away the moment the model reaches the GPU. Converting the
files once on disk means each cold start reads ~39 GB instead of ~77 GB, which roughly
halves the time a fresh worker needs before it can answer your first request.

The VAE and the LoRA files are left alone - they are small and not worth the risk.

Run it on a RunPod Pod with the network volume attached (at least 32 GB of system RAM):

    python convert_to_bf16.py /workspace/weights/LongCat-Video

Safe to interrupt and re-run: each shard is written to a temporary file first and only
then swapped in, and shards that are already bfloat16 are skipped.
"""

import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Only these subfolders are worth converting (they are the float32 heavyweights).
FOLDERS_TO_CONVERT = ["dit", "text_encoder"]


def human(num_bytes):
    return f"{num_bytes / 1024 ** 3:.1f} GB"


def already_bfloat16(path):
    """Peek at the file header instead of loading gigabytes into RAM."""
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            dtype = handle.get_slice(key).get_dtype()
            if dtype in ("F32", "F16", "F64"):
                return False
    return True


def convert_shard(path):
    """Rewrite one .safetensors file as bfloat16. Returns the new size in bytes."""
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}

    tensors = load_file(path, device="cpu")
    converted = {}
    total_bytes = 0
    for key, tensor in tensors.items():
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        converted[key] = tensor
        total_bytes += tensor.numel() * tensor.element_size()
    del tensors

    metadata.setdefault("format", "pt")
    tmp_path = path + ".converting"
    save_file(converted, tmp_path, metadata=metadata)
    del converted

    os.replace(tmp_path, path)  # atomic: the old file is never half-overwritten
    return total_bytes


def update_index(folder, byte_counts):
    """Keep the *.safetensors.index.json 'total_size' field honest."""
    for name in os.listdir(folder):
        if not name.endswith(".index.json"):
            continue
        index_path = os.path.join(folder, name)
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        if "metadata" not in index:
            index["metadata"] = {}
        index["metadata"]["total_size"] = sum(byte_counts.values())
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
        print(f"    updated {name}")


def convert_folder(folder):
    shards = sorted(f for f in os.listdir(folder) if f.endswith(".safetensors"))
    if not shards:
        print(f"  no .safetensors files in {folder}, skipping")
        return

    byte_counts = {}
    for shard in shards:
        path = os.path.join(folder, shard)
        before = os.path.getsize(path)

        if already_bfloat16(path):
            print(f"  {shard}: already bfloat16 ({human(before)}), skipping")
            byte_counts[shard] = before
            continue

        print(f"  {shard}: converting ({human(before)}) ...", flush=True)
        byte_counts[shard] = convert_shard(path)
        print(f"  {shard}: {human(before)} -> {human(os.path.getsize(path))}", flush=True)

    update_index(folder, byte_counts)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    if not os.path.isdir(checkpoint_dir):
        print(f"error: {checkpoint_dir} is not a folder")
        sys.exit(1)

    for name in FOLDERS_TO_CONVERT:
        folder = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(folder):
            print(f"warning: {folder} not found, skipping")
            continue
        print(f"\n== {name} ==")
        convert_folder(folder)

    print("\nConversion finished.")


if __name__ == "__main__":
    main()
