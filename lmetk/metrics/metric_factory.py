"""Factory for creating metric instances."""

from typing import Dict, Type
from .base import AbstractMetric
from lmetk.metrics.metrics import MAEMetric, MSEMetric, RMSEMetric, MAPEMetric, MSPEMetric, RSEMetric, CORRMetric


class MetricFactory:
    """Factory class for creating metric instances."""
    
    _metrics: Dict[str, Type[AbstractMetric]] = {
        'mae': MAEMetric,
        'mse': MSEMetric,
        'rmse': RMSEMetric,
        'mape': MAPEMetric,
        'mspe': MSPEMetric,
        'rse': RSEMetric,
        'corr': CORRMetric,
    }
    
    @classmethod
    def create_metric(cls, metric_name: str) -> AbstractMetric:
        """Create a metric instance by name."""
        metric_name_lower = metric_name.lower()
        if metric_name_lower not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")
        
        return cls._metrics[metric_name_lower]()
    
    @classmethod
    def get_available_metrics(cls) -> list:
        """Get list of available metric names."""
        return list(cls._metrics.keys())