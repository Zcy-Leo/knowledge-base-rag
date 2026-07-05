import os
from huggingface_hub import hf_hub_download

model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cache_dir = "C:/Users/HP/.cache/huggingface/hub"

files_to_download = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json"
]

print(f"Downloading model: {model_id}")
print(f"Cache directory: {cache_dir}")
print("=" * 60)

for filename in files_to_download:
    try:
        print(f"Downloading {filename}...")
        filepath = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=True
        )
        print(f"  ✓ {filename} -> {filepath}")
    except Exception as e:
        print(f"  ✗ {filename} failed: {e}")

print("=" * 60)
print("Download completed!")
