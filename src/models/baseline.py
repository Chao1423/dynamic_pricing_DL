"""
Baseline models: Linear, ElasticNet, RandomForest.
"""
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression


class LinearModel:
    """
    Linear regression baseline.
    """
    
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.model.get_params()


class ElasticNetModel:
    """
    ElasticNet regression baseline.
    """
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.model.get_params()


class RandomForestModel:
    """
    Random Forest regression baseline.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.model.get_params()
