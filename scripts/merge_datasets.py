#!/usr/bin/env python
"""
Dataset Merger Script

Merges multiple car parts datasets into a unified format for YOLO training.

Supports:
- carparts-seg (legacy ID-based remap)
- car-parts-1 (legacy offset-based remap)
- Any Roboflow dataset (class-name-based automatic remap)

Usage:
    python scripts/merge_datasets.py
"""

import shutil
from pathlib import Path

import yaml


# Common class name aliases for cross-dataset matching
CLASS_ALIASES = {
    "back_bumper": "rear_bumper",
    "back_door": "rear_door",
    "back_glass": "rear_windshield",
    "back_left_door": "rear_door",
    "back_left_light": "rear_light",
    "back_light": "rear_light",
    "back_right_door": "rear_door",
    "back_right_light": "rear_light",
    "front_glass": "front_windshield",
    "front_left_door": "front_door",
    "front_left_light": "front_light",
    "front_right_door": "front_door",
    "front_right_light": "front_light",
    "left_mirror": "mirror",
    "right_mirror": "mirror",
    "tailgate": "trunk",
    "headlamp": "headlights",
    "head_light": "headlights",
    "taillamp": "taillights",
    "tail_light": "taillights",
    "brake-pad": "brake_pad",
    "brake_pad": "brake_pad",
    "brake-disc": "brake_rotor",
    "brake_disc": "brake_rotor",
    "brake-caliper": "brake_caliper",
    "brake_caliper": "brake_caliper",
    "brake-rotor": "brake_rotor",
    "brake_rotor": "brake_rotor",
    "spark-plug": "spark_plug",
    "spark_plug": "spark_plug",
    "oil-filter": "oil_filter",
    "oil_filter": "oil_filter",
    "air-filter": "air_filter",
    "air_filter": "air_filter",
    "fuel-filter": "fuel_filter",
    "fuel_filter": "fuel_filter",
    "cabin-filter": "cabin_filter",
    "cabin_filter": "cabin_filter",
    "wiper-blade": "wiper_blade",
    "wiper_blade": "wiper_blade",
    "window-regulator": "window_regulator",
    "window_regulator": "window_regulator",
    "water-pump": "water_pump",
    "water_pump": "water_pump",
    "radiator-hose": "radiator_hose",
    "radiator_hose": "radiator_hose",
    "radiator-fan": "radiator_fan",
    "radiator_fan": "radiator_fan",
    "oil-pan": "oil_pan",
    "oil_pan": "oil_pan",
    "oil-pressure-sensor": "oil_pressure_sensor",
    "oil_pressure_sensor": "oil_pressure_sensor",
    "oxygen-sensor": "oxygen_sensor",
    "oxygen_sensor": "oxygen_sensor",
    "ignition-coil": "ignition_coil",
    "ignition_coil": "ignition_coil",
    "fuel-injector": "fuel_injector",
    "fuel_injector": "fuel_injector",
    "engine-valve": "engine_valve",
    "engine_valve": "engine_valve",
    "engine-block": "engine_block",
    "engine_block": "engine_block",
    "lower-control-arm": "lower_control_arm",
    "lower_control_arm": "lower_control_arm",
    "leaf-spring": "leaf_spring",
    "leaf_spring": "leaf_spring",
    "coil-spring": "coil_spring",
    "coil_spring": "coil_spring",
    "clutch-plate": "clutch_plate",
    "clutch_plate": "clutch_plate",
    "cylinder-head": "cylinder_head",
    "cylinder_head": "cylinder_head",
    "pressure-plate": "pressure_plate",
    "pressure_plate": "pressure_plate",
    "valve-lifter": "valve_lifter",
    "valve_lifter": "valve_lifter",
    "vacuum-brake-booster": "vacuum_brake_booster",
    "vacuum_brake_booster": "vacuum_brake_booster",
    "instrument-cluster": "instrument_cluster",
    "instrument_cluster": "instrument_cluster",
    "fuse-box": "fuse_box",
    "fuse_box": "fuse_box",
    "gas-cap": "gas_cap",
    "gas_cap": "gas_cap",
    "idler-arm": "idler_arm",
    "idler_arm": "idler_arm",
    "distributor": "distributor",
    "camshaft": "camshaft",
    "crankshaft": "crankshaft",
    "piston": "piston",
    "muffler": "muffler",
    "alternator": "alternator",
    "starter": "starter",
    "transmission": "transmission",
    "torque-converter": "torque_converter",
    "torque_converter": "torque_converter",
    "carburetor": "carberator",
    "carberator": "carberator",
    "carb": "carberator",
    "rim": "rim",
    "shift-knob": "shift_knob",
    "shift_knob": "shift_knob",
    "spoiler": "spoiler",
    "side-mirror": "side_mirror",
    "side_mirror": "side_mirror",
    "radio": "radio",
    "overflow-tank": "overflow_tank",
    "overflow_tank": "overflow_tank",
    "thermostat": "thermostat",
    "air-compressor": "air_compressor",
    "air_compressor": "air_compressor",
    "wheel": "wheel",
    "hood": "hood",
    "trunk": "trunk",
    "grille": "grille",
    "battery": "battery",
    "radiator": "radiator",
    # New dataset aliases
    "damaged_spark_plug": "spark_plug",
    "spark-plug-damaged": "spark_plug",
    "brake-disk": "brake_rotor",
    "brake_shoes": "brake_pad",
    "brake-shoes": "brake_pad",
    "brake-shoe": "brake_pad",
    "clutch-lever": "clutch_plate",
    "clutch_lever": "clutch_plate",
    "clutch": "clutch_plate",
    "front_brake": "brake_pad",
    "front-brake": "brake_pad",
    "rear_brake": "brake_pad",
    "rear-brake": "brake_pad",
    "gasoline_filter": "fuel_injector",
    "gasoline-filter": "fuel_injector",
    "motor": "starter",
    "rearview_mirror": "side_mirror",
    "rearview-mirror": "side_mirror",
    "front_sprocket": "chain",
    "front-sprocket": "chain",
    "crown": "engine_block",
    "retainers": "valve_lifter",
    # Filter out garbage class names
    "d008": None,
    "d228": None,
    "isimsiz": None,
}


class DatasetMerger:
    """Merge multiple datasets into a unified YOLO format."""

    def __init__(self, output_dir: str = "ml_datasets/combined_car_parts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        self.train_count = 0
        self.val_count = 0

        # Load existing classes from current data.yaml, or use defaults
        self.unified_classes = self._load_existing_classes()

        # Legacy mappings (kept for backward compatibility)
        self.carparts_seg_mapping = {
            0: 1, 1: 5, 2: 9, 3: 5, 4: 7, 5: 7, 6: 5, 7: 7,
            8: 0, 9: 4, 10: 8, 11: 4, 12: 6, 13: 6, 14: 4, 15: 6,
            16: 2, 17: 11, 18: -1, 19: 11, 20: 3, 21: 3, 22: 10,
        }
        # car-parts-1 explicit class mapping (class ID → unified class ID)
        # Extracted from combined_car_parts/data.yaml positions 13-62
        self.car_parts_1_mapping = {
            0: 13, 1: 14, 2: 15, 3: 16, 4: 17, 5: 18, 6: 19, 7: 20, 8: 21, 9: 22,
            10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 29, 17: 30, 18: 31, 19: 32,
            20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 28: 41, 29: 42,
            30: 43, 31: 44, 32: 45, 33: 46, 34: 47, 35: 48, 36: 49, 37: 50, 38: 51, 39: 52,
            40: 53, 41: 54, 42: 55, 43: 56, 44: 57, 45: 58, 46: 59, 47: 60, 48: 61, 49: 62,
        }

    def _load_existing_classes(self) -> list[str]:
        """Load existing class names from data.yaml if it exists."""
        yaml_path = self.output_dir / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, dict):
                    max_idx = max(names.keys()) if names else -1
                    classes = [""] * (max_idx + 1)
                    for idx, name in names.items():
                        classes[int(idx)] = name
                    return classes
                elif isinstance(names, list):
                    return names
            except Exception as e:
                print(f"  Warning: Could not read existing data.yaml: {e}")

        # Default fallback classes (from current dataset)
        return [
            "front_bumper", "rear_bumper", "hood", "trunk", "front_door",
            "rear_door", "front_light", "rear_light", "front_windshield",
            "rear_windshield", "wheel", "mirror", "grille", "window_regulator",
            "water_pump", "thermostat", "taillights", "starter",
            "torque_converter", "radio", "radiator_hose", "overflow_tank",
            "radiator_fan", "oil_filter", "oil_pressure_sensor", "valve_lifter",
            "oxygen_sensor", "oil_pan", "radiator", "pressure_plate", "piston",
            "muffler", "lower_control_arm", "instrument_cluster", "gas_cap",
            "engine_valve", "vacuum_brake_booster", "fuse_box", "headlights",
            "ignition_coil", "fuel_injector", "idler_arm", "leaf_spring",
            "engine_block", "distributor", "camshaft", "brake_pad",
            "transmission", "brake_rotor", "carberator", "crankshaft",
            "coil_spring", "brake_caliper", "clutch_plate", "cylinder_head",
            "battery", "air_compressor", "alternator", "rim", "spark_plug",
            "side_mirror", "shift_knob", "spoiler",
        ]

    def _normalize_class_name(self, name: str) -> str | None:
        """Normalize class name for matching."""
        normalized = name.lower().strip().replace(" ", "_").replace("-", "_")
        # Apply aliases (None means skip/garbage)
        return CLASS_ALIASES.get(normalized, normalized)

    def _build_class_mapping(self, source_classes: list[str]) -> dict[int, int]:
        """Build mapping from source class IDs to unified class IDs."""
        mapping = {}
        unified_normalized = {
            self._normalize_class_name(c): i
            for i, c in enumerate(self.unified_classes)
        }

        for src_id, src_name in enumerate(source_classes):
            normalized = self._normalize_class_name(src_name)
            # Skip garbage classes mapped to None
            if normalized is None:
                continue
            if normalized in unified_normalized:
                mapping[src_id] = unified_normalized[normalized]
            else:
                # Add as new class
                new_id = len(self.unified_classes)
                self.unified_classes.append(normalized)
                unified_normalized[normalized] = new_id
                mapping[src_id] = new_id
                print(f"    New class added: {normalized} (ID {new_id})")

        return mapping

    def merge_roboflow_dataset(self, dataset_dir: str, prefix: str = ""):
        """
        Merge any Roboflow/YOLO-format dataset by class name matching.

        Args:
            dataset_dir: Path to dataset (contains data.yaml, train/, valid/, test/)
            prefix: Filename prefix for merged files
        """
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            print(f"  Dataset not found at {dataset_dir}")
            return 0

        # Read source data.yaml
        yaml_path = dataset_dir / "data.yaml"
        if not yaml_path.exists():
            yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            print(f"  No data.yaml found in {dataset_dir}")
            return 0

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        source_names = data.get("names", {})
        if isinstance(source_names, dict):
            max_idx = max(source_names.keys()) if source_names else -1
            source_classes = [""] * (max_idx + 1)
            for idx, name in source_names.items():
                source_classes[int(idx)] = name
        else:
            source_classes = source_names

        print(f"\nMerging {dataset_dir.name}...")
        print(f"  Source classes: {len(source_classes)}")

        # Build class mapping
        class_mapping = self._build_class_mapping(source_classes)
        print(f"  Mapped {len(class_mapping)} classes")

        count = 0
        for split in ["train", "valid", "test"]:
            img_split_dir = dataset_dir / split / "images"
            lbl_split_dir = dataset_dir / split / "labels"

            if not img_split_dir.exists():
                # Some datasets use different structure
                img_split_dir = dataset_dir / "images" / split
                lbl_split_dir = dataset_dir / "labels" / split

            if not img_split_dir.exists():
                continue

            target_split = "val" if split in ["valid", "test"] else "train"

            for img_file in img_split_dir.glob("*.*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                fname = f"{prefix}_{img_file.name}" if prefix else img_file.name
                img_dest = self.output_dir / "images" / target_split / fname
                shutil.copy2(img_file, img_dest)

                label_file = lbl_split_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_dest = self.output_dir / "labels" / target_split / f"{Path(fname).stem}.txt"
                    self._remap_labels_by_mapping(label_file, label_dest, class_mapping)

                count += 1
                if target_split == "train":
                    self.train_count += 1
                else:
                    self.val_count += 1

        print(f"  Added {count} images from {dataset_dir.name}")
        return count

    def merge_carparts_seg(self, images_dir: str, labels_dir: str):
        """Merge carparts-seg dataset (legacy)."""
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        if not images_dir.exists():
            print(f"  Carparts-seg images not found at {images_dir}")
            return 0

        print("Merging carparts-seg...")
        count = 0

        for split in ["train", "val", "test"]:
            img_split_dir = images_dir / split
            lbl_split_dir = labels_dir / split

            if not img_split_dir.exists():
                continue

            target_split = "val" if split in ["val", "test"] else "train"

            for img_file in img_split_dir.glob("*.*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                img_dest = self.output_dir / "images" / target_split / f"carparts_{img_file.name}"
                shutil.copy2(img_file, img_dest)

                label_file = lbl_split_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_dest = self.output_dir / "labels" / target_split / f"carparts_{img_file.stem}.txt"
                    self._remap_labels_carparts_legacy(label_file, label_dest)

                count += 1
                if target_split == "train":
                    self.train_count += 1
                else:
                    self.val_count += 1

        print(f"  Added {count} images from carparts-seg")
        return count

    def merge_car_parts_1(self, dataset_dir: str):
        """Merge car-parts-1 (Roboflow) dataset (legacy offset method)."""
        dataset_dir = Path(dataset_dir)

        if not dataset_dir.exists():
            print(f"  car-parts-1 not found at {dataset_dir}")
            return 0

        print("Merging car-parts-1 (legacy offset)...")
        count = 0

        for split in ["train", "valid", "test"]:
            img_split_dir = dataset_dir / split / "images"
            lbl_split_dir = dataset_dir / split / "labels"

            if not img_split_dir.exists():
                continue

            target_split = "val" if split in ["valid", "test"] else "train"

            for img_file in img_split_dir.glob("*.*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                img_dest = self.output_dir / "images" / target_split / f"parts1_{img_file.name}"
                shutil.copy2(img_file, img_dest)

                label_file = lbl_split_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    label_dest = self.output_dir / "labels" / target_split / f"parts1_{img_file.stem}.txt"
                    self._remap_labels_by_mapping(label_file, label_dest, self.car_parts_1_mapping)

                count += 1
                if target_split == "train":
                    self.train_count += 1
                else:
                    self.val_count += 1

        print(f"  Added {count} images from car-parts-1")
        return count

    def _remap_labels_by_mapping(self, src_file: Path, dest_file: Path, mapping: dict[int, int]):
        """Remap labels using a class ID mapping."""
        with open(src_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id in mapping:
                parts[0] = str(mapping[class_id])
                new_lines.append(' '.join(parts) + '\n')

        with open(dest_file, 'w') as f:
            f.writelines(new_lines)

    def _remap_labels_carparts_legacy(self, src_file: Path, dest_file: Path):
        """Legacy carparts-seg remapping."""
        with open(src_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id in self.carparts_seg_mapping:
                new_class_id = self.carparts_seg_mapping[class_id]
                if new_class_id >= 0:
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts) + '\n')

        with open(dest_file, 'w') as f:
            f.writelines(new_lines)

    def _offset_labels(self, src_file: Path, dest_file: Path, offset: int):
        """Add offset to all class IDs."""
        with open(src_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            parts[0] = str(class_id + offset)
            new_lines.append(' '.join(parts) + '\n')

        with open(dest_file, 'w') as f:
            f.writelines(new_lines)

    def create_data_yaml(self):
        """Create data.yaml for the combined dataset."""
        yaml_content = f"""# Combined Car Parts Dataset
# Generated by merge_datasets.py
# Total classes: {len(self.unified_classes)}

path: ./ml_datasets/combined_car_parts
train: images/train
val: images/val

nc: {len(self.unified_classes)}
names:
"""
        for i, name in enumerate(self.unified_classes):
            yaml_content += f"  {i}: {name}\n"

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"\nCreated: {yaml_path}")

    def print_summary(self):
        """Print merge summary."""
        print("\n" + "="*50)
        print("   MERGE COMPLETE")
        print("="*50)
        print(f"  Train images: {self.train_count}")
        print(f"  Val images: {self.val_count}")
        print(f"  Total: {self.train_count + self.val_count}")
        print(f"  Classes: {len(self.unified_classes)}")
        print(f"\nOutput directory: {self.output_dir}")


def main():
    print("="*50)
    print("   DATASET MERGER")
    print("="*50)
    print()

    merger = DatasetMerger()

    # Merge legacy datasets
    merger.merge_carparts_seg("ml_datasets/images", "ml_datasets/labels")
    merger.merge_car_parts_1("ml_datasets/car-parts-1")

    # Merge new Roboflow datasets (auto class name matching)
    new_datasets = [
        ("ml_datasets/car-parts-ybiev", "robo1"),
        ("ml_datasets/brake-pad-vpcfl", "robo2"),
        ("ml_datasets/spark_plug", "robo3"),
        ("ml_datasets/machine-parts-bwqjl", "robo4"),
    ]

    for ds_path, prefix in new_datasets:
        if Path(ds_path).exists():
            merger.merge_roboflow_dataset(ds_path, prefix)
        else:
            print(f"\n  Skipping {ds_path} (not downloaded)")

    # Create data.yaml
    merger.create_data_yaml()

    # Print summary
    merger.print_summary()

    print("\n" + "="*50)
    print("   NEXT STEPS")
    print("="*50)
    print("  1. Verify dataset: ls ml_datasets/combined_car_parts/images/train")
    print("  2. Train model:")
    print("     python scripts/train_model.py")


if __name__ == "__main__":
    main()
