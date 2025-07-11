"""Concrete metric implementations."""

import numpy as np
from .base import AbstractFileSystemMetric, AbstractMetric
import pandas as pd

class NormalizedMAEMetricRailway(AbstractFileSystemMetric):
    """Normalized Mean Absolute Error metric for Railvay."""
    
    def __init__(self, paths: dict = None):
        super().__init__("mae", paths)
        self.factor = pd.read_csv(f"{paths.data_dir}/railway/railway.csv")["Bahnlast"].abs().mean()
    
    def reset(self):
        self.values = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        abs_errors = (np.mean(np.abs(predictions - targets)) * 100) / self.factor
        self.values.extend(abs_errors.flatten())
    
    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)

class NormalizedMSEMetricRailway(AbstractFileSystemMetric):
    """"Normalized Mean Squared Error metric for Railvay."""
    
    def __init__(self, paths: dict = None):
        super().__init__("mse", paths)
        self.factor = pd.read_csv(f"{paths.data_dir}/railway/railway.csv")["Bahnlast"].abs().mean()
    
    def reset(self):
        self.values = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        squared_errors = (np.mean((predictions - targets) ** 2) * 100) / self.factor
        self.values.extend(squared_errors.flatten())
    
    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)



class MAEMetric(AbstractMetric):
    """Mean Absolute Error metric."""
    
    def __init__(self):
        super().__init__("mae")
    
    def reset(self):
        self.values = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        abs_errors = np.mean(np.abs(predictions - targets))
        self.values.extend(abs_errors.flatten())
    
    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)



# class MSEMetric(AbstractMetric):
#     """Mean Squared Error metric."""
    
#     def __init__(self):
#         super().__init__("mse")
    
#     def reset(self):
#         self.values = []
    
#     def update(self, predictions: np.ndarray, targets: np.ndarray):
#         squared_errors = np.mean((predictions - targets) ** 2) 
#         self.values.extend(squared_errors.flatten())
    
#     def compute(self) -> float:
#         if not self.values:
#             raise ValueError("No data available for computation")
#         return np.mean(self.values)

class MSEMetric(AbstractMetric):
    """Mean Squared Error metric with batch-wise accumulation for (B, L, 1) shapes."""
    
    def __init__(self):
        super().__init__("mse")
        self.reset()
    
    def reset(self):
        self.total_squared_error = 0.0
        self.total_count = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Ensure shapes are compatible
        assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"
        
        # Compute squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Accumulate
        self.total_squared_error += np.sum(squared_errors)
        self.total_count += predictions.shape[0] * predictions.shape[1]  # B * L
    
    def compute(self) -> float:
        if self.total_count == 0:
            raise ValueError("No data available for computation")
        return self.total_squared_error / self.total_count

class RMSEMetric(AbstractMetric):
    """Root Mean Squared Error metric with batch-wise accumulation for (B, L, 1) shapes."""
    
    def __init__(self):
        super().__init__("rmse")
        self.reset()
    
    def reset(self):
        self.total_squared_error = 0.0
        self.total_count = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Ensure shapes match
        assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"

        # Compute squared errors
        squared_errors = (predictions - targets) ** 2

        # Accumulate sum and count
        self.total_squared_error += np.sum(squared_errors)
        self.total_count += np.prod(targets.shape)  # B * L * 1
    
    def compute(self) -> float:
        if self.total_count == 0:
            raise ValueError("No data available for computation")
        return np.sqrt(self.total_squared_error / self.total_count)

# class RMSEMetric(AbstractMetric):
#     """Root Mean Squared Error metric."""
    
#     def __init__(self):
#         super().__init__("rmse")
    
#     def reset(self):
#         self.values = []
    
#     def update(self, predictions: np.ndarray, targets: np.ndarray):
#         squared_errors = np.sqrt(np.mean((predictions - targets) ** 2))
#         self.values.extend(squared_errors.flatten())
    
#     def compute(self) -> float:
#         if not self.values:
#             raise ValueError("No data available for computation")
#         return np.mean(self.values)


class MAPEMetric(AbstractMetric):
    """Mean Absolute Percentage Error metric."""
    
    def __init__(self):
        super().__init__("mape")
    
    def reset(self):
        self.values = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Avoid division by zero
        mask = targets != 0
        if np.any(mask):
            percentage_errors = np.abs((predictions[mask] - targets[mask]) / targets[mask])
            self.values.extend(percentage_errors.flatten())
    
    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)


class MSPEMetric(AbstractMetric):
    """Mean Squared Percentage Error metric."""
    
    def __init__(self):
        super().__init__("mspe")
    
    def reset(self):
        self.values = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Avoid division by zero
        mask = targets != 0
        if np.any(mask):
            percentage_errors = ((predictions[mask] - targets[mask]) / targets[mask]) ** 2
            self.values.extend(percentage_errors.flatten())
    
    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)


class RSEMetric(AbstractMetric):
    """Relative Squared Error metric."""
    
    def __init__(self):
        super().__init__("rse")
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
    
    def compute(self) -> float:
        if not self.predictions:
            raise ValueError("No data available for computation")
        
        pred = np.array(self.predictions)
        true = np.array(self.targets)
        
        numerator = np.sqrt(np.sum((true - pred) ** 2))
        denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
        
        return numerator / denominator


class CORRMetric(AbstractMetric):
    """Correlation coefficient metric."""
    
    def __init__(self):
        super().__init__("corr")
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
    
    def compute(self) -> float:
        if not self.predictions:
            raise ValueError("No data available for computation")
        
        pred = np.array(self.predictions)
        true = np.array(self.targets)
        
        u = ((true - true.mean()) * (pred - pred.mean())).sum()
        d = np.sqrt(((true - true.mean()) ** 2 * (pred - pred.mean()) ** 2).sum())
        
        return u / d if d != 0 else 0.0