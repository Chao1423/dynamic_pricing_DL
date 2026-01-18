"""
Dataset classes for loading and batching preprocessed data.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AirlineDataset(Dataset):
    """
    PyTorch Dataset for airline ticket price prediction.
    Handles categorical and numerical features separately.
    """
    
    def __init__(
        self,
        data_path: str,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: str = 'price_log',
        cat_maps_path: Optional[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to processed parquet/csv file
            cat_cols: List of categorical column names
            num_cols: List of numerical column names
            target_col: Target column name
            cat_maps_path: Path to cat_maps.json (for validation)
        """
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        
        # Load data
        if data_path.endswith('.parquet'):
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        
        # Load category mappings if provided
        self.cat_maps = None
        if cat_maps_path:
            with open(cat_maps_path, 'r') as f:
                self.cat_maps = json.load(f)
        
        # Extract features and target
        self.cat_features = self.df[cat_cols].values.astype(np.int64)
        self.num_features = self.df[num_cols].values.astype(np.float32)
        self.targets = self.df[target_col].values.astype(np.float32)
        
        # Store original price if available
        if 'price_original' in self.df.columns:
            self.price_original = self.df['price_original'].values.astype(np.float32)
        else:
            self.price_original = None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'cat', 'num', 'target' keys
        """
        return {
            'cat': torch.LongTensor(self.cat_features[idx]),
            'num': torch.FloatTensor(self.num_features[idx]),
            'target': torch.FloatTensor([self.targets[idx]])
        }
    
    def get_cardinalities(self) -> List[int]:
        """
        Get cardinalities for each categorical feature.
        
        Returns:
            List of cardinalities (including UNK)
        """
        if self.cat_maps:
            return [len(mapping) for mapping in self.cat_maps.values()]
        else:
            # Infer from data (max value + 1 to include UNK=0)
            return [int(self.cat_features[:, i].max()) + 1 for i in range(len(self.cat_cols))]


def load_datasets(
    data_dir: str,
    cat_cols: List[str],
    num_cols: List[str],
    target_col: str = 'price_log',
    cat_maps_path: Optional[str] = None
) -> Tuple[AirlineDataset, AirlineDataset, AirlineDataset]:
    """
    Load train, validation, and test datasets.
    
    Args:
        data_dir: Directory containing processed data files
        cat_cols: List of categorical column names
        num_cols: List of numerical column names
        target_col: Target column name
        cat_maps_path: Path to cat_maps.json
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)
    
    if cat_maps_path is None:
        cat_maps_path = data_dir / 'cat_maps.json'
    
    # Try parquet first, fallback to CSV
    def get_data_path(name: str) -> str:
        parquet_path = data_dir / f'{name}.parquet'
        csv_path = data_dir / f'{name}.csv'
        if parquet_path.exists():
            return str(parquet_path)
        elif csv_path.exists():
            return str(csv_path)
        else:
            raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists")
    
    train_ds = AirlineDataset(
        get_data_path('train'),
        cat_cols, num_cols, target_col, cat_maps_path
    )
    val_ds = AirlineDataset(
        get_data_path('val'),
        cat_cols, num_cols, target_col, cat_maps_path
    )
    test_ds = AirlineDataset(
        get_data_path('test'),
        cat_cols, num_cols, target_col, cat_maps_path
    )
    
    return train_ds, val_ds, test_ds
