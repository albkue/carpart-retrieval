# Set up the "ml" conda env for the car-part retrieval project.
# Run from PowerShell, from the repo root, with your conda base activated:
#   conda activate ml
#   .\scripts\setup_env.ps1
#
# Order matters. torch and paddlepaddle-gpu need CUDA-specific package
# indexes; everything else in requirements.txt is generic PyPI. Installing
# requirements.txt first means pip silently satisfies the torch dependency
# of ultralytics/open-clip-torch/FlagEmbedding with a CPU wheel, and you get
# no error -- just a model that trains 10x slower and never touches the GPU.

param(
    [string]$CudaTag = "cu126"   # match your driver: cu118 | cu126 | cu128 (torch), cu118 | cu126 | cu129 (paddle)
)

Write-Host "== 1/4: checking GPU / driver ==" -ForegroundColor Cyan
nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Warning "nvidia-smi failed -- no NVIDIA driver visible in this shell. Stop and fix that before continuing, or re-run with CPU-only intent."
    Read-Host "Press Enter to continue anyway, or Ctrl+C to stop"
}

Write-Host "== 2/4: PyTorch (CUDA: $CudaTag) ==" -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CudaTag"

Write-Host "== 3/4: PaddlePaddle-GPU (CUDA: $CudaTag) ==" -ForegroundColor Cyan
python -m pip install "paddlepaddle-gpu==3.2.2" -i "https://www.paddlepaddle.org.cn/packages/stable/$CudaTag/"

Write-Host "== 4/4: everything else (requirements.txt) ==" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "== verifying ==" -ForegroundColor Cyan
python scripts\check_env.py
