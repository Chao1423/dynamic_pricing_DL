"""
Plot training curves from saved JSON files.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(curves_path: Path, output_path: Path):
    """Plot training curves from JSON file."""
    with open(curves_path, 'r') as f:
        curves = json.load(f)
    
    epochs = range(1, len(curves['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(epochs, curves['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs, curves['val_loss'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MAE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # MAE curves
    axes[1].plot(epochs, curves['train_mae'], label='Train MAE', color='blue')
    axes[1].plot(epochs, curves['val_mae'], label='Val MAE', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--curves', type=str, required=True, help='Path to curves JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    
    args = parser.parse_args()
    
    plot_training_curves(Path(args.curves), Path(args.output))


if __name__ == '__main__':
    main()
