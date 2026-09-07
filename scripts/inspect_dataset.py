#!/usr/bin/env python
"""Inspect combined_car_parts dataset — sample images from each class."""

import shutil
import random
from pathlib import Path
import yaml


def inspect_dataset(
    base_path: str = r"C:\Users\HP Omen 16\.vscode\fix3\ml_service\ml_datasets\combined_car_parts",
    output_path: str = r"C:\Users\HP Omen 16\.vscode\fix3\ml_service\ml_datasets\combined_car_parts\temp_inspect",
    samples_per_class: int = 10,
):
    """Sample random images for each class and copy to inspection folder."""
    base = Path(base_path)
    label_dir = base / "labels/train"
    image_dir = base / "images/train"
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read class names from data.yaml
    yaml_path = base / "data.yaml"
    class_names = []
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            names = data.get("names", {})
            if isinstance(names, dict):
                class_names = [names[i] for i in sorted(names.keys())]
            elif isinstance(names, list):
                class_names = names
    num_classes = len(class_names) if class_names else 66

    # Build class-to-image mapping
    class_images = {i: [] for i in range(num_classes)}

    print("Scanning label files...")
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            found_classes = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    found_classes.add(int(parts[0]))
            for cls in found_classes:
                if cls in class_images:
                    class_images[cls].append(label_file.stem)

    # Print summary
    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)
    for cls_id in sorted(class_images.keys()):
        count = len(class_images[cls_id])
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        if count > 0:
            print(f"  Class {cls_id:2d} ({name:20s}): {count:5d} images")

    # Sample and copy images for inspection
    print("\n" + "=" * 50)
    print(f"SAMPLING {samples_per_class} IMAGES PER CLASS")
    print("=" * 50)

    for cls_id in sorted(class_images.keys()):
        images = class_images[cls_id]
        if not images:
            continue

        # Pick random samples
        picks = random.sample(images, min(samples_per_class, len(images)))

        # Create output folder
        folder_name = f"class_{cls_id}"
        part_folder = output_dir / folder_name
        part_folder.mkdir(exist_ok=True)

        copied = 0
        for stem in picks:
            for ext in [".jpg", ".jpeg", ".png"]:
                src = image_dir / (stem + ext)
                if src.exists():
                    dst = part_folder / f"{stem}{ext}"
                    shutil.copy2(src, dst)
                    copied += 1
                    break

        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  Class {cls_id:2d} ({name:20s}): {copied}/{samples_per_class} sampled")

    print(f"\nInspection complete. View samples at: {output_dir.resolve()}")


if __name__ == "__main__":
    inspect_dataset()