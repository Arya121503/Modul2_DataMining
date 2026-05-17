# This file makes the src directory a Python package

from .data_loader import DataLoader
from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model import ModelBuilder
from .evaluation import ModelEvaluator

__all__ = [
    'DataLoader',
    'DataPreprocessor', 
    'FeatureEngineer',
    'ModelBuilder',
    'ModelEvaluator'
]

__version__ = '1.0.0'
__author__ = 'Data Mining Student'
