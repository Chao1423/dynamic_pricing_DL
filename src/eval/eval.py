"""
Evaluation script for trained models.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.datasets.datasets import load_datasets
from src.models.baseline import ElasticNetModel, LinearModel, RandomForestModel
from src.models.deep import MLPWithEmbeddings, TabTransformer
from src.utils.metrics import compute_metrics
from src.utils.utils import set_seed


def evaluate_baseline(
    model_path: str,
    test_data: Dict,
    model_name: str
) -> Dict:
    """
    Evaluate baseline model.
    
    Args:
        model_path: Path to saved model
        test_data: Dict with 'X', 'y', 'y_original' keys
        model_name: Model name
        
    Returns:
        Dictionary with metrics
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    test_pred = model.predict(test_data['X'])
    
    # Compute metrics
    metrics = compute_metrics(
        test_data['y'], test_pred, test_data.get('y_original')
    )
    
    return metrics


def evaluate_deep(
    model_path: str,
    test_loader: DataLoader,
    cat_cardinalities: list,
    num_features: int,
    model_name: str,
    device: torch.device
) -> Dict:
    """
    Evaluate deep learning model.
    
    Args:
        model_path: Path to saved model state dict
        test_loader: Test data loader
        cat_cardinalities: List of categorical cardinalities
        num_features: Number of numerical features
        model_name: Model name
        device: Device to evaluate on
        
    Returns:
        Dictionary with metrics
    """
    # Initialize model
    if model_name == 'mlp_embeddings':
        model = MLPWithEmbeddings(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features
        )
    elif model_name == 'tabtransformer':
        model = TabTransformer(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features
        )
    else:
        raise ValueError(f"Unknown deep model: {model_name}")
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluate
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            cat = batch['cat'].to(device)
            num = batch['num'].to(device)
            target = batch['target'].to(device)
            
            pred = model(cat, num)
            test_preds.append(pred.cpu().numpy())
            test_targets.append(target.cpu().numpy())
    
    test_preds = np.concatenate(test_preds).flatten()
    test_targets = np.concatenate(test_targets).flatten()
    
    # Get original prices if available
    test_original = None
    if hasattr(test_loader.dataset, 'price_original') and test_loader.dataset.price_original is not None:
        test_original = test_loader.dataset.price_original
    
    # Compute metrics
    metrics = compute_metrics(test_targets, test_preds, test_original)
    
    return metrics


def prepare_baseline_data(dataset):
    """Prepare data for baseline models."""
    X_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        X_list.append(torch.cat([sample['cat'].float(), sample['num']]).numpy())
    X = np.array(X_list)
    y = dataset.targets
    y_original = dataset.price_original if dataset.price_original is not None else None
    return {'X': X, 'y': y, 'y_original': y_original}


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--model', type=str, required=True,
                       choices=['linear', 'elasticnet', 'randomforest',
                               'mlp_embeddings', 'tabtransformer'],
                       help='Model to evaluate')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('random_seed', 42))
    
    # Load data
    data_dir = config['data']['processed_dir']
    cat_cols = config['features']['categorical']
    num_cols = config['features']['numerical']
    
    _, _, test_ds = load_datasets(
        data_dir, cat_cols, num_cols,
        cat_maps_path=Path(data_dir) / 'cat_maps.json'
    )
    
    # Evaluate
    if args.model in ['linear', 'elasticnet', 'randomforest']:
        test_data = prepare_baseline_data(test_ds)
        metrics = evaluate_baseline(args.model_path, test_data, args.model)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        test_loader = DataLoader(
            test_ds, batch_size=config['training']['batch_size']
        )
        
        cat_cardinalities = test_ds.get_cardinalities()
        num_features = len(num_cols)
        
        metrics = evaluate_deep(
            args.model_path, test_loader,
            cat_cardinalities, num_features, args.model, device
        )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f'{args.model}_test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Test Metrics for {args.model}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
