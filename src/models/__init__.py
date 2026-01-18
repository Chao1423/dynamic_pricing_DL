"""
Model definitions for airline ticket price prediction.
"""
from .baseline import LinearModel, ElasticNetModel, RandomForestModel
from .deep import MLPWithEmbeddings, TabTransformer

__all__ = [
    'LinearModel',
    'ElasticNetModel',
    'RandomForestModel',
    'MLPWithEmbeddings',
    'TabTransformer'
]
