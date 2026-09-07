#!/usr/bin/env python
"""
Download Motor Oil Dataset from Roboflow

This script downloads motor oil detection datasets for product recognition.

Datasets:
- Motor Oil (343 images): https://universe.roboflow.com/car-oil/motor-oil
- Engine Oil Detection (185 images): https://universe.roboflow.com/newproject-7c1cc/engine-oil-detection

Usage:
    cd ml_service
    python scripts/download_motor_oil.py
"""

import zipfile
import requests
from pathlib import Path


def download_from_url(url: str, output_path: str):
    """Download file from URL with progress."""
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"\r  Progress: {percent:.1f}%", end='')
    
    print("\n  Done!")
    return True


def extract_zip(zip_path: str, extract_to: str):
    """Extract ZIP file."""
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("  Done!")
    return True


def main():
    print("="*60)
    print("   MOTOR OIL DATASET DOWNLOAD")
    print("="*60)
    print()
    
    # Create output directory
    output_dir = Path("ml_datasets/motor_oil")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  MANUAL DOWNLOAD REQUIRED                                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  Roboflow requires manual download via browser.          ║
    ║                                                          ║
    ║  Steps:                                                  ║
    ║  1. Create account at https://app.roboflow.com (free)    ║
    ║  2. Go to:                                               ║
    ║     https://universe.roboflow.com/car-oil/motor-oil      ║
    ║                                                          ║
    ║  3. Click "Fork" to copy to your workspace               ║
    ║  4. Go to your project                                   ║
    ║  5. Click "Export Dataset"                               ║
    ║  6. Select "YOLOv8" format                               ║
    ║  7. Click "Download" to get ZIP                          ║
    ║                                                          ║
    ║  8. Extract ZIP to:                                      ║
    ║     ml_service/ml_datasets/motor_oil/                       ║
    ║                                                          ║
    ║  Expected structure:                                     ║
    ║  ml_datasets/motor_oil/                                     ║
    ║  ├── train/                                              ║
    ║  │   ├── images/                                         ║
    ║  │   └── labels/                                         ║
    ║  ├── valid/                                              ║
    ║  │   ├── images/                                         ║
    ║  │   └── labels/                                         ║
    ║  └── data.yaml                                           ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if already downloaded
    expected_dir = Path("ml_datasets/motor_oil/train")
    if expected_dir.exists():
        print("✓ Motor oil dataset already exists!")
        print("  Location: ml_datasets/motor_oil/")
        
        # Count images
        train_images = list((Path("ml_datasets/motor_oil/train/images")).glob("*.*"))
        print(f"  Training images: {len(train_images)}")
        
        return
    
    print("\nWaiting for manual download...")
    print("After downloading, run: python scripts/merge_datasets.py")


if __name__ == "__main__":
    main()
