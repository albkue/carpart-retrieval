#!/usr/bin/env python
"""
Car Parts Dataset Download Script

Downloads recommended Roboflow datasets to expand training data.
Datasets:
- team-data/car-parts-ybiev (8,739 images, pre-trained model)
- waste/brake-pad-vpcfl (1,638 brake pad images)
- royalewithcheese/spark_plug (247 spark plug images)
- penguin/machine-parts-bwqjl (1,000+ machine parts)

Usage:
    export ROBOFLOW_API_KEY="your_key"
    python scripts/download_datasets.py

Get API key: https://app.roboflow.com/settings/api
"""

import os
import subprocess
import zipfile
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Dataset configuration
ML_DATASETS_DIR = Path("ml_datasets")
ML_DATASETS_DIR.mkdir(exist_ok=True)

# Recommended Roboflow datasets
ROBOFLOW_DATASETS = [
    {
        "workspace": "team-data",
        "project": "car-parts-ybiev",
        "version": 1,
        "desc": "Car Parts (8,739 images, pre-trained)",
    },
    {
        "workspace": "waste-t3zae",
        "project": "brake-pad-vpcfl",
        "version": 2,
        "desc": "Brake Pad v2 (1,638 images)",
    },
    {
        "workspace": "royalewithcheese",
        "project": "spark_plug",
        "version": 1,
        "desc": "Spark Plug (247 images)",
    },
    {
        "workspace": "penguin-wxcxp",
        "project": "machine-parts-bwqjl",
        "version": 1,
        "desc": "Machine Parts (1,000+ images)",
    },
]


def run_command(cmd, cwd=None):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def download_with_roboflow(workspace: str, project: str, version: int, desc: str) -> bool:
    """Download a dataset from Roboflow using the Python API."""
    print("\n" + "="*50)
    print(f"Downloading: {desc}")
    print("="*50)

    target_dir = ML_DATASETS_DIR / f"{project}"
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"  {target_dir} already exists. Skipping...")
        return True

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("  ERROR: ROBOFLOW_API_KEY not set!")
        print("  Get your key at: https://app.roboflow.com/settings/api")
        print("  Then run: export ROBOFLOW_API_KEY='your_key'")
        return False

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        proj.version(version).download("yolov8", location=str(target_dir))
        print(f"  Downloaded to: {target_dir}")
        return True
    except Exception as e:
        print(f"  ERROR downloading {project}: {e}")
        print("  Manual alternative:")
        print(f"    1. Visit: https://universe.roboflow.com/{workspace}/{project}")
        print("    2. Click 'Download' → YOLOv8 format")
        print(f"    3. Extract to: {target_dir}/")
        return False


def download_compcars():
    """Download CompCars dataset from Kaggle."""
    print("\n" + "="*50)
    print("Downloading CompCars Dataset...")
    print("="*50)

    compcars_dir = ML_DATASETS_DIR / "compcars"
    if compcars_dir.exists():
        print("CompCars already exists. Skipping...")
        return True

    cmd = "kaggle datasets download -d renancostaalencar/compcars"
    if not run_command(cmd, cwd=str(ML_DATASETS_DIR)):
        print("Failed to download CompCars. Make sure Kaggle CLI is configured.")
        return False

    zip_path = ML_DATASETS_DIR / "compcars.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(compcars_dir)
        zip_path.unlink()
        print(f"Extracted CompCars to {compcars_dir}")

    return True


def download_carparts_seg():
    """Download Carparts-seg dataset from Ultralytics."""
    print("\n" + "="*50)
    print("Downloading Carparts-seg Dataset...")
    print("="*50)

    carparts_dir = ML_DATASETS_DIR / "carparts-seg"
    if carparts_dir.exists():
        print("Carparts-seg already exists. Skipping...")
        return True

    try:
        import urllib.request

        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
        zip_path = ML_DATASETS_DIR / "carparts-seg.zip"

        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(carparts_dir)
        zip_path.unlink()
        print(f"Extracted Carparts-seg to {carparts_dir}")

        return True
    except Exception as e:
        print(f"Error downloading Carparts-seg: {e}")
        return False


def download_stanford_cars():
    """Download Stanford Cars dataset from Kaggle."""
    print("\n" + "="*50)
    print("Downloading Stanford Cars Dataset...")
    print("="*50)

    stanford_dir = ML_DATASETS_DIR / "stanford_cars"
    if stanford_dir.exists():
        print("Stanford Cars already exists. Skipping...")
        return True

    cmd = "kaggle datasets download -d eduardo4jesus/stanford-cars-dataset"
    if not run_command(cmd, cwd=str(ML_DATASETS_DIR)):
        print("Failed to download Stanford Cars.")
        return False

    zip_path = ML_DATASETS_DIR / "stanford-cars-dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(stanford_dir)
        zip_path.unlink()
        print(f"Extracted Stanford Cars to {stanford_dir}")

    return True


def main():
    """Main download function."""
    print("="*60)
    print("   CAR PARTS DATASET DOWNLOAD SCRIPT")
    print("="*60)

    has_api_key = bool(os.environ.get("ROBOFLOW_API_KEY"))

    print("\nDatasets to download:")
    print("  [Roboflow - Auto]")
    for ds in ROBOFLOW_DATASETS:
        print(f"    • {ds['desc']}")
    print("  [Kaggle/Ultralytics - Optional]")
    print("    • CompCars (27,618 images)")
    print("    • Carparts-seg (3,833 images)")
    print("    • Stanford Cars (16,185 images)")

    if not has_api_key:
        print("\n" + "!"*60)
        print("  WARNING: ROBOFLOW_API_KEY not set!")
        print("  Roboflow datasets will be skipped.")
        print("  Get key: https://app.roboflow.com/settings/api")
        print("!"*60)

    print("\nPress Enter to start (or Ctrl+C to cancel)...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    results = {}

    # Download Roboflow datasets
    if has_api_key:
        for ds in ROBOFLOW_DATASETS:
            success = download_with_roboflow(
                ds["workspace"], ds["project"], ds["version"], ds["desc"]
            )
            results[ds["desc"]] = success
    else:
        print("\nSkipping Roboflow datasets (no API key).")

    # Optional: Download Kaggle/Ultralytics datasets
    print("\nDownload Kaggle/Ultralytics datasets too? (y/n): ", end="")
    try:
        ans = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"

    if ans == "y":
        results["CompCars"] = download_compcars()
        results["Carparts-seg"] = download_carparts_seg()
        results["Stanford Cars"] = download_stanford_cars()

    # Summary
    print("\n" + "="*60)
    print("   DOWNLOAD SUMMARY")
    print("="*60)

    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {dataset}: {status}")

    print("\nNext steps:")
    print("  1. Run merge script:")
    print("     python scripts/merge_datasets.py")
    print("  2. Train model:")
    print("     python scripts/train_model.py")


if __name__ == "__main__":
    main()
