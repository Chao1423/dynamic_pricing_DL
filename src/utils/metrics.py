"""
Evaluation metrics for regression tasks.
"""
from typing import Optional

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_original: Optional[np.ndarray] = None
) -> dict:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values (log space)
        y_pred: Predicted values (log space)
        y_original: Original values (original space, optional)
        
    Returns:
        Dictionary of metrics
    """
    # Log space metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    # Original space metrics (if available)
    if y_original is not None:
        y_pred_original = np.expm1(y_pred)  # Inverse log1p
        metrics['rmse_original'] = float(np.sqrt(np.mean((y_original - y_pred_original) ** 2)))
        metrics['mae_original'] = float(np.mean(np.abs(y_original - y_pred_original)))
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_original != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_original[mask] - y_pred_original[mask]) / y_original[mask])) * 100
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = float('inf')
    
    return metrics
