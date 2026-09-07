#!/usr/bin/env python
"""
Car Parts Detection Model Training Script

This script trains YOLOv8 models on car parts datasets.

Usage:
    # Train with Carparts-seg (auto-download)
    python scripts/train_model.py --dataset carparts-seg
    
    # Train with custom dataset
    python scripts/train_model.py --dataset custom --data path/to/data.yaml
    
    # Train with specific model size
    python scripts/train_model.py --model yolov8m --epochs 100
"""

import argparse
import os

def train_model(args):
    """Train the YOLO model."""
    from ultralytics import YOLO
    
    # Determine model size
    model_sizes = {
        'n': 'yolov8n.pt',   # Nano - fastest, least accurate
        's': 'yolov8s.pt',   # Small
        'm': 'yolov8m.pt',   # Medium - balanced
        'l': 'yolov8l.pt',   # Large
        'x': 'yolov8x.pt',   # Extra large - most accurate
    }
    
    model_name = model_sizes.get(args.model, f'yolov8{args.model}.pt')
    print(f"Using model: {model_name}")
    
    # Load model (resume from checkpoint if requested)
    if args.resume:
        # Search for last.pt in possible locations (handles nested project paths)
        possible_paths = [
            f"{args.project}/{args.name}/weights/last.pt",
            f"{args.project}/{args.project}/{args.name}/weights/last.pt",
        ]
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find last.pt in any of: {possible_paths}"
            )
        print(f"Resuming from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        model = YOLO(model_name)
    
    # Dataset configuration
    if args.data:
        data_config = args.data
    elif args.dataset == 'carparts-seg':
        # Built-in dataset
        data_config = 'carparts-seg.yaml'
    elif args.dataset == 'custom':
        data_config = 'ml_datasets/combined_car_parts/data.yaml'
    else:
        data_config = f'{args.dataset}.yaml'
    
    print(f"Dataset: {data_config}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    
    # Train
    results = model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        
        # Learning rate and optimizer
        lr0=args.lr,              # Initial learning rate
        lrf=0.01,                 # Final learning rate (lr0 * lrf)
        momentum=0.937,           # SGD momentum/Adam beta1
        weight_decay=0.0005,      # Optimizer weight decay
        optimizer=args.optimizer, # Optimizer: SGD, Adam, AdamW
        
        # Learning rate scheduler
        cos_lr=True,              # Cosine learning rate scheduler
        warmup_epochs=3.0,        # Warmup epochs
        warmup_momentum=0.8,      # Warmup momentum
        warmup_bias_lr=0.1,       # Warmup bias learning rate
        
        # Augmentation settings for robustness
        mosaic=1.0,           # Mosaic augmentation
        mixup=0.15,           # MixUp augmentation
        copy_paste=0.1,       # Copy-paste augmentation
        degrees=15.0,         # Rotation augmentation
        translate=0.1,        # Translation augmentation
        scale=0.9,            # Scale augmentation
        fliplr=0.5,           # Horizontal flip
        hsv_h=0.015,          # HSV-Hue augmentation
        hsv_s=0.7,            # HSV-Saturation augmentation
        hsv_v=0.4,            # HSV-Value augmentation
        
        # Training settings
        patience=20,          # Early stopping patience
        save=True,            # Save checkpoints
        save_period=10,       # Save every N epochs
        
        # Output
        project=args.project, 
        name=args.name,
        
        # Performance
        device=args.device,   # GPU device (0, 1, 2, ... or 'cpu')
        workers=args.workers,
        
        # Additional settings
        verbose=True,         # Verbose output
        exist_ok=True,        # Overwrite existing project
        resume=args.resume,   # Resume from checkpoint
    )
    
    print("\n" + "="*60)
    print("   TRAINING COMPLETE")
    print("="*60)
    
    # Validate
    print("\nRunning validation...")
    metrics = model.val()
    
    print("\nResults:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    # Export
    if args.export:
        print("\nExporting model...")
        export_path = model.export(format='onnx')
        print(f"Exported to: {export_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train car parts detection model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='carparts-seg',
                       choices=['carparts-seg', 'custom', 'coco', 'voc'],
                       help='Dataset to use for training')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to custom data.yaml file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='m',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (0, 1, 2, ... or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loader workers')
    
    # Learning rate and optimizer arguments
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='auto',
                       choices=['SGD', 'Adam', 'AdamW', 'auto'],
                       help='Optimizer type (default: auto - SGD with warmup)')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory for outputs')
    parser.add_argument('--name', type=str, default='car_parts_v1',
                       help='Experiment name')
    
    # Export
    parser.add_argument('--export', action='store_true',
                       help='Export model to ONNX after training')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    print("="*60)
    print("   CAR PARTS DETECTION MODEL TRAINING")
    print("="*60)
    
    train_model(args)


if __name__ == '__main__':
    main()
