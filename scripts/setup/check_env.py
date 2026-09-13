#!/usr/bin/env python3
"""Verify the ml env has everything this project needs, and that the GPU
builds actually landed (not silently downgraded to CPU by pip). Run after
scripts/setup/setup_env.ps1, and any time you're not sure the env is sane.

    python scripts/setup/check_env.py
"""
import importlib
import sys

CHECKS = [
    "torch", "torchvision", "ultralytics", "open_clip", "FlagEmbedding",
    "faiss", "paddleocr", "paddle", "fastapi", "uvicorn", "sqlalchemy",
    "psycopg2", "alembic", "cv2", "PIL", "imagehash", "yaml", "mlflow", "dvc",
]

failed = []
for mod in CHECKS:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  ok   {mod:<14} {v}")
    except Exception as e:
        failed.append(mod)
        print(f"  FAIL {mod:<14} {e}")

print()
try:
    import torch
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  torch built for CUDA: {torch.version.cuda}")
    else:
        print("  !! torch has NO CUDA support -- you likely installed requirements.txt "
              "before running setup_env.ps1's torch step, and pip gave you the CPU "
              "wheel. Reinstall: pip uninstall torch torchvision torchaudio -y, then "
              "re-run scripts/setup/setup_env.ps1 in order.")
except ImportError:
    pass

try:
    import paddle
    print(f"paddle.is_compiled_with_cuda(): {paddle.is_compiled_with_cuda()}")
except ImportError:
    pass

if failed:
    print(f"\n{len(failed)} package(s) missing: {', '.join(failed)}")
    sys.exit(1)
print("\nAll checks passed.")
