"""
Deep learning models: MLP with Embeddings, TabTransformer.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPWithEmbeddings(nn.Module):
    """
    MLP with embedding layers for categorical features.
    Enhanced with BatchNorm and improved structure.
    """
    
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_features: int,
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        output_dim: int = 1
    ):
        """
        Initialize MLP with embeddings.
        
        Args:
            cat_cardinalities: List of cardinalities for each categorical feature
            num_features: Number of numerical features
            embedding_dim: Embedding dimension for categoricals
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            use_batchnorm: Whether to use BatchNorm
            output_dim: Output dimension
        """
        super().__init__()
        
        # Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim) for card in cat_cardinalities
        ])
        
        # Input dimension: sum of embeddings + numerical features
        input_dim = len(cat_cardinalities) * embedding_dim + num_features
        
        # MLP layers with BatchNorm
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cat: Categorical features [batch_size, num_cat]
            num: Numerical features [batch_size, num_num]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        # Embed categoricals
        cat_embeds = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_concat = torch.cat(cat_embeds, dim=1)
        
        # Concatenate with numericals
        x = torch.cat([cat_concat, num], dim=1)
        
        # MLP
        return self.mlp(x)


class TabTransformer(nn.Module):
    """
    TabTransformer model for tabular data.
    Uses transformer encoder for categoricals, MLP for numericals.
    Enhanced with BatchNorm and improved structure.
    """
    
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_features: int,
        embedding_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        output_dim: int = 1
    ):
        """
        Initialize TabTransformer.
        
        Args:
            cat_cardinalities: List of cardinalities for each categorical feature
            num_features: Number of numerical features
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension in transformer
            dropout: Dropout rate
            use_batchnorm: Whether to use BatchNorm
            output_dim: Output dimension
        """
        super().__init__()
        
        # Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim) for card in cat_cardinalities
        ])
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Numerical feature processing
        self.num_proj = nn.Linear(num_features, embedding_dim)
        
        # Output head with BatchNorm
        # Input: cat_pooled (embedding_dim) + num_proj (embedding_dim) = 2 * embedding_dim
        self.output_dim = output_dim
        output_layers = [
            nn.Linear(2 * embedding_dim, hidden_dim)
        ]
        if use_batchnorm:
            output_layers.append(nn.BatchNorm1d(hidden_dim))
        output_layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        ])
        self.output_head = nn.Sequential(*output_layers)
    
    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cat: Categorical features [batch_size, num_cat]
            num: Numerical features [batch_size, num_num]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        batch_size = cat.size(0)
        
        # Embed categoricals
        cat_embeds = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        # Stack: [batch_size, num_cat, embedding_dim]
        cat_stack = torch.stack(cat_embeds, dim=1)
        
        # Transformer on categoricals
        cat_transformed = self.transformer(cat_stack)
        # Pool: mean over categorical features
        cat_pooled = cat_transformed.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Project numericals
        num_proj = self.num_proj(num)  # [batch_size, embedding_dim]
        
        # Concatenate and output
        x = torch.cat([cat_pooled, num_proj], dim=1)
        return self.output_head(x)
