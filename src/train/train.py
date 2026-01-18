"""
Training script for airline ticket price prediction models.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.datasets.datasets import load_datasets
from src.models.baseline import ElasticNetModel, LinearModel, RandomForestModel
from src.models.deep import MLPWithEmbeddings, TabTransformer
from src.utils.metrics import compute_metrics
from src.utils.utils import set_seed


def train_baseline(
    model_name: str,
    train_data: Dict,
    val_data: Dict,
    config: Dict
) -> Dict:
    """
    Train baseline model (Linear/ElasticNet/RandomForest).
    
    Args:
        model_name: Model name ('linear', 'elasticnet', 'randomforest')
        train_data: Dict with 'X', 'y' keys
        val_data: Dict with 'X', 'y' keys
        config: Model configuration
        
    Returns:
        Dictionary with model and metrics
    """
    # Initialize model
    if model_name == 'linear':
        model = LinearModel(**config.get('params', {}))
    elif model_name == 'elasticnet':
        model = ElasticNetModel(**config.get('params', {}))
    elif model_name == 'randomforest':
        model = RandomForestModel(**config.get('params', {}))
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
    
    # Train
    print(f"Training {model_name}...")
    model.fit(train_data['X'], train_data['y'])
    
    # Evaluate
    train_pred = model.predict(train_data['X'])
    val_pred = model.predict(val_data['X'])
    
    train_metrics = compute_metrics(train_data['y'], train_pred, train_data.get('y_original'))
    val_metrics = compute_metrics(val_data['y'], val_pred, val_data.get('y_original'))
    
    print(f"Train RMSE: {train_metrics['rmse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }


def train_deep(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cat_cardinalities: list,
    num_features: int,
    config: Dict,
    device: torch.device
) -> Dict:
    """
    Train deep learning model.
    
    Args:
        model_name: Model name ('mlp_embeddings', 'tabtransformer')
        train_loader: Training data loader
        val_loader: Validation data loader
        cat_cardinalities: List of categorical cardinalities
        num_features: Number of numerical features
        config: Model configuration
        device: Device to train on
        
    Returns:
        Dictionary with model and metrics
    """
    # Initialize model
    if model_name == 'mlp_embeddings':
        model = MLPWithEmbeddings(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            **config.get('params', {})
        )
    elif model_name == 'tabtransformer':
        model = TabTransformer(
            cat_cardinalities=cat_cardinalities,
            num_features=num_features,
            **config.get('params', {})
        )
    else:
        raise ValueError(f"Unknown deep model: {model_name}")
    
    model = model.to(device)
    
    # Optimizer and loss (use MAE/L1Loss)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    criterion = nn.L1Loss()  # MAE loss
    
    # Training loop with early stopping
    num_epochs = config.get('num_epochs', 100)
    early_stop_patience = config.get('early_stop_patience', 10)
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training curves
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_preds_list = []
        train_targets_list = []
        for batch in train_loader:
            cat = batch['cat'].to(device)
            num = batch['num'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            pred = model(cat, num)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds_list.append(pred.detach().cpu().numpy())
            train_targets_list.append(target.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_preds_epoch = np.concatenate(train_preds_list).flatten()
        train_targets_epoch = np.concatenate(train_targets_list).flatten()
        train_mae = np.mean(np.abs(train_targets_epoch - train_preds_epoch))
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                cat = batch['cat'].to(device)
                num = batch['num'].to(device)
                target = batch['target'].to(device)
                
                pred = model(cat, num)
                loss = criterion(pred, target)
                val_loss += loss.item()
                
                val_preds.append(pred.cpu().numpy())
                val_targets.append(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds_epoch = np.concatenate(val_preds).flatten()
        val_targets_epoch = np.concatenate(val_targets).flatten()
        val_mae = np.mean(np.abs(val_targets_epoch - val_preds_epoch))
        
        # Record curves
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        # Early stopping based on validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Patience: {patience_counter}/{early_stop_patience}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Return training curves
    training_curves = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_mae': train_maes,
        'val_mae': val_maes
    }
    
    # Final evaluation
    model.eval()
    train_preds = []
    train_targets = []
    with torch.no_grad():
        for batch in train_loader:
            cat = batch['cat'].to(device)
            num = batch['num'].to(device)
            target = batch['target'].to(device)
            pred = model(cat, num)
            train_preds.append(pred.cpu().numpy())
            train_targets.append(target.cpu().numpy())
    
    train_preds = np.concatenate(train_preds).flatten()
    train_targets = np.concatenate(train_targets).flatten()
    
    # Re-evaluate validation set with best model
    val_preds_final = []
    val_targets_final = []
    with torch.no_grad():
        for batch in val_loader:
            cat = batch['cat'].to(device)
            num = batch['num'].to(device)
            target = batch['target'].to(device)
            pred = model(cat, num)
            val_preds_final.append(pred.cpu().numpy())
            val_targets_final.append(target.cpu().numpy())
    
    val_preds_final = np.concatenate(val_preds_final).flatten()
    val_targets_final = np.concatenate(val_targets_final).flatten()
    
    train_metrics = compute_metrics(train_targets, train_preds)
    val_metrics = compute_metrics(val_targets_final, val_preds_final)
    
    print(f"Train RMSE: {train_metrics['rmse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_curves': training_curves
    }


def prepare_baseline_data(dataset):
    """Prepare data for baseline models (concatenate features)."""
    X_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        X_list.append(torch.cat([sample['cat'].float(), sample['num']]).numpy())
    X = np.array(X_list)
    y = dataset.targets
    y_original = dataset.price_original if dataset.price_original is not None else None
    return {'X': X, 'y': y, 'y_original': y_original}


def main():
    parser = argparse.ArgumentParser(description='Train airline ticket price models')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['linear', 'elasticnet', 'randomforest', 
                               'mlp_embeddings', 'tabtransformer'],
                       help='Model to train')
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
    
    train_ds, val_ds, test_ds = load_datasets(
        data_dir, cat_cols, num_cols,
        cat_maps_path=Path(data_dir) / 'cat_maps.json'
    )
    
    # Train model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model in ['linear', 'elasticnet', 'randomforest']:
        # Baseline models
        train_data = prepare_baseline_data(train_ds)
        val_data = prepare_baseline_data(val_ds)
        
        model_config = config['models'][args.model]
        results = train_baseline(args.model, train_data, val_data, model_config)
        
        # Save model
        with open(output_dir / f'{args.model}_model.pkl', 'wb') as f:
            pickle.dump(results['model'], f)
    
    else:
        # Deep models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        train_loader = DataLoader(
            train_ds, batch_size=config['training']['batch_size'], shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])
        
        cat_cardinalities = train_ds.get_cardinalities()
        num_features = len(num_cols)
        
        model_config = config['models'][args.model]
        results = train_deep(
            args.model, train_loader, val_loader,
            cat_cardinalities, num_features, model_config, device
        )
        
        # Save model
        torch.save(results['model'].state_dict(), output_dir / f'{args.model}_model.pt')
        
        # Save training curves if available
        if 'training_curves' in results:
            # Convert numpy types to Python native types for JSON serialization
            curves = results['training_curves']
            curves_serializable = {
                'train_loss': [float(x) for x in curves['train_loss']],
                'val_loss': [float(x) for x in curves['val_loss']],
                'train_mae': [float(x) for x in curves['train_mae']],
                'val_mae': [float(x) for x in curves['val_mae']]
            }
            with open(output_dir / f'{args.model}_curves.json', 'w') as f:
                json.dump(curves_serializable, f, indent=2)
    
    # Save metrics
    with open(output_dir / f'{args.model}_metrics.json', 'w') as f:
        json.dump({
            'train': results['train_metrics'],
            'val': results['val_metrics']
        }, f, indent=2)
    
    print(f"Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
