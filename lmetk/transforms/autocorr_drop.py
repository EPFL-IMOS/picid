import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Union

from lmetk.transforms.base_transform import BaseTransform 


class AutocorrFeatureDropTransform(BaseTransform):

    def __init__(self, correlation_threshold: float = 0.95, method: str = 'pearson', 
                 keep_first: bool = True, exclude_columns: Optional[List[int]] = None, **kwargs):
        """
        Initialize the autocorrelation feature drop transform.
        
        Args:
            correlation_threshold: Threshold above which features are considered highly correlated (default: 0.95)
            method: Correlation method ('pearson', 'spearman', 'kendall') (default: 'pearson')
            keep_first: If True, keep the first feature in correlated pairs; if False, keep the last (default: True)
            exclude_columns: List of column indices to exclude from correlation analysis (always keep these)
            **kwargs: Additional keyword arguments
        """
        self._init_kwargs = kwargs 
        self.correlation_threshold = correlation_threshold
        self.method = method
        self.keep_first = keep_first
        self.exclude_columns = exclude_columns or []
        self.features_to_drop = None
        self.features_to_keep = None
        self.is_fitted = False

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        main_args = f"correlation_threshold={self.correlation_threshold!r}, method={self.method!r}, keep_first={self.keep_first!r}, exclude_columns={self.exclude_columns!r}"
        if args:
            return f"{self.__class__.__name__}({main_args}, {args})"
        return f"{self.__class__.__name__}({main_args})"

    def fit_single_source(self, data_segment: Any, mode: str):
        """
        Fit the transform by identifying highly correlated features to drop.
        
        Args:
            data_segment: The data segment to analyze for correlations
        """
        # Convert to numpy array if needed
        if not isinstance(data_segment, np.ndarray):
            data_segment = np.array(data_segment)
        
        # Ensure 2D
        if data_segment.ndim == 1:
            data_segment = data_segment.reshape(-1, 1)
        
        # Convert to DataFrame for correlation analysis
        df = pd.DataFrame(data_segment)
        
        # Calculate correlation matrix
        if self.method == 'pearson':
            corr_matrix = df.corr(method='pearson')
        elif self.method == 'spearman':
            corr_matrix = df.corr(method='spearman')
        elif self.method == 'kendall':
            corr_matrix = df.corr(method='kendall')
        else:
            raise ValueError(f"Unsupported correlation method: {self.method}. Use 'pearson', 'spearman', or 'kendall'")
        
        # Find highly correlated feature pairs
        features_to_drop = set()
        n_features = corr_matrix.shape[0]
        
        # Get upper triangle of correlation matrix (excluding diagonal)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Skip if either feature is in exclude list
                if i in self.exclude_columns or j in self.exclude_columns:
                    continue
                
                # Check if correlation exceeds threshold
                if abs(corr_matrix.iloc[i, j]) >= self.correlation_threshold:
                    # Decide which feature to drop
                    if self.keep_first:
                        features_to_drop.add(j)  # Drop the later feature
                    else:
                        features_to_drop.add(i)  # Drop the earlier feature
        
        # Store the features to drop and keep
        self.features_to_drop = sorted(list(features_to_drop))
        self.features_to_keep = [i for i in range(n_features) if i not in features_to_drop]
        self.is_fitted = True
        
        print(f"Correlation analysis complete:")
        print(f"  - Total features: {n_features}")
        print(f"  - Features to drop: {len(self.features_to_drop)} {self.features_to_drop}")
        print(f"  - Features to keep: {len(self.features_to_keep)}")
        print(f"  - Correlation threshold: {self.correlation_threshold}")
        print(f"  - Method: {self.method}")

    def fit_multi_source(self, list_of_data_segments: List[Any],  mode: str):
        """
        Fit the transform using multiple data segments.
        Concatenates all segments for correlation analysis.
        
        Args:
            list_of_data_segments: List of data segments to analyze
        """
        if not list_of_data_segments:
            raise ValueError("list_of_data_segments cannot be empty")
        
        # Convert all segments to numpy arrays and ensure 2D
        processed_segments = []
        for segment in list_of_data_segments:
            if not isinstance(segment, np.ndarray):
                segment = np.array(segment)
            if segment.ndim == 1:
                segment = segment.reshape(-1, 1)
            processed_segments.append(segment)
        
        # Ensure all segments have the same number of features
        n_features = processed_segments[0].shape[1]
        for i, segment in enumerate(processed_segments):
            if segment.shape[1] != n_features:
                raise ValueError(f"Segment {i} has {segment.shape[1]} features, expected {n_features}")
        
        # Concatenate all segments
        combined_data = np.concatenate(processed_segments, axis=0)
        
        print(f"Fitting on {len(list_of_data_segments)} segments with combined shape: {combined_data.shape}")
        
        # Fit on combined data
        self.fit_single_source(combined_data)

    def transform_data(self, data_segment: Any) -> Any:
        """
        Transform data by dropping highly correlated features.
        
        Args:
            data_segment: The data segment to transform
            
        Returns:
            Data segment with correlated features removed
        """
        if not self.is_fitted:
            raise ValueError("Transform must be fitted before calling transform_data. Call fit_single_source or fit_multi_source first.")
        
        # Convert to numpy array if needed
        if not isinstance(data_segment, np.ndarray):
            data_segment = np.array(data_segment)
        
        # Ensure 2D
        if data_segment.ndim == 1:
            data_segment = data_segment.reshape(-1, 1)
        
        # Validate number of features matches what we fitted on
        if data_segment.shape[1] != len(self.features_to_keep) + len(self.features_to_drop):
            raise ValueError(f"Data segment has {data_segment.shape[1]} features, "
                           f"but transform was fitted on {len(self.features_to_keep) + len(self.features_to_drop)} features")
        
        # Keep only the selected features
        result = data_segment[:, self.features_to_keep]
        
        print(f"Transformed data from shape {data_segment.shape} to {result.shape}")
        
        return result

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about which features were kept/dropped.
        
        Returns:
            Dictionary with feature selection information
        """
        if not self.is_fitted:
            return {"error": "Transform not fitted yet"}
        
        return {
            "features_kept": self.features_to_keep,
            "features_dropped": self.features_to_drop,
            "n_features_original": len(self.features_to_keep) + len(self.features_to_drop),
            "n_features_final": len(self.features_to_keep),
            "reduction_ratio": len(self.features_to_drop) / (len(self.features_to_keep) + len(self.features_to_drop)),
            "correlation_threshold": self.correlation_threshold,
            "method": self.method
        }

    def __call__(self, data_segment: Any) -> Any:
        return self.transform_data(data_segment)