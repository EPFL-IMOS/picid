from typing import Dict, Any, List, Optional, Union

# To allow instantiation of Hydra-configured objects

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

# Import your strategy and base transform classes
from lmetk.transforms.base_transform import BaseTransform
from lmetk.transforms.strategy import (
    TransformStrategy,
    # DirectTransformStrategy,
    # FitTransformStrategy,
    # MultiSourceFitTransformStrategy,
    # MultiSourceDirectTransformStrategy
    UnifiedDirectTransformStrategy,
    UnifiedFitTransformStrategy,
)
from lmetk.transforms.multisource import BaseMultiSourceTransformMixin

import logging
logger = logging.getLogger(__name__) 

class DataTransform:
    """
    Main transform class that coordinates different transformation strategies.
    It acts as a facade, deciding which strategy to use based on provided metadata
    and then delegating the actual transformation application. It also handles
    post-transformation validation.
    """
    
    def __init__(self, transform_name: str, transform: dict, metadata: dict):
        """
        Initializes DataTransform with a name, the transform configuration, and metadata.

        Parameters
        ----------
        transform_name : str
            A unique name for this transformation.
        transform : dict
            A dictionary containing the configuration for the actual transform
            instance (e.g., scikit-learn scaler), typically to be instantiated by Hydra.
            This instantiated object must implement `BaseTransform`.
        metadata : dict
            A dictionary containing metadata that guides how the transform should be applied.
            Expected keys include:
            - 'apply_to': str (required) - The key in the data dictionaries where the transform applies (e.g., 'features').
            - 'fit_on': str (optional) - The split to fit the transform on (e.g., 'train').
            - 'multi_source_fit': bool (optional, default False) - If True, the transform
              will be fitted across multiple sources, requiring a list of data dictionaries
              as input to its `forward` method.
        """
        self.transform_name = transform_name
        self.metadata = metadata
        
        self.apply_to = metadata.get('apply_to', None)
        self.fit_on = metadata.get('fit_on', None)
        self.auxiliary_keys = metadata.get('auxiliary_keys', None)
        self.multi_source_fit = metadata.get('multi_source_fit', False)

        if self.apply_to is None:
            raise ValueError(f"Metadata for transform '{transform_name}' must specify 'apply_to' key.")
        
        assert isinstance(transform, BaseTransform), (
                f"Transform instance for '{transform_name}' (type: {type(transform)}) "
                "must implement BaseTransform."
            )
        
        if self.fit_on is not None:
            self.strategy = UnifiedFitTransformStrategy()
        else:
            self.strategy = UnifiedDirectTransformStrategy()

        # Infer the strategy based on metadata flags
        # TODO: this logic might be a bit broken
        # if isinstance(transform, BaseTransform) and isinstance(transform, BaseMultiSourceTransformMixin):
        #     if self.fit_on is not None:
        #         self.strategy = MultiSourceFitTransformStrategy()
        #     else:
        #         self.strategy = MultiSourceDirectTransformStrategy()
        
        # else:
        #     assert isinstance(transform, BaseTransform), (
        #         f"Transform instance for '{transform_name}' (type: {type(transform)}) "
        #         "must implement BaseTransform."
        #     )
        #     # Determine the strategy based on the presence of 'fit_on'
        #     if  self.fit_on is not None:
        #         self.strategy = FitTransformStrategy()
        #     else:
        #         self.strategy = DirectTransformStrategy()
            
        # Initialize the actual transform instance
        # Ensure the instantiated object implements BaseTransform
        if isinstance(transform, DictConfig):
            self.transform_instance: BaseTransform = instantiate(transform)
        else:
            self.transform_instance = transform
            assert isinstance(self.transform_instance, BaseTransform), (
                f"Transform instance for '{transform_name}' (type: {type(self.transform_instance)}) "
                f"must inherit from 'lmetk.transforms.base_transform.BaseTransform'."
            )
        
        if not isinstance(self.transform_instance, BaseTransform):
            raise TypeError(
                f"Transform instance for '{transform_name}' (type: {type(self.transform_instance)}) "
                f"must inherit from 'lmetk.transforms.base_transform.BaseTransform'."
            )
            
    
    def forward(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Applies the transformation to the input data using the selected strategy.
        Performs validation checks after the strategy returns the transformed data.

        Parameters
        ----------
        data : Union[Dict[str, Any], List[Dict[str, Any]]]
            Input data. This can be:
            - A single dictionary: for single-source transformations (DirectTransformStrategy, FitTransformStrategy).
              Expected structure: `{ 'data_key': { 'train': ..., 'val': ..., 'test': ... } }`
            - A list of dictionaries: for multi-source transformations (MultiSourceFitTransformStrategy).
              Expected structure: `[ { 'data_key': { ... } }, { 'data_key': { ... } } ]`
            
        Returns
        -------
        Union[Dict[str, Any], List[Dict[str, Any]]]
            Transformed data, in the same structure as the input.

        Raises
        ------
        TypeError
            If the strategy returns an unexpected data type, or if keys in transformed
            data don't hold dictionaries of splits.
        KeyError
            If there's a mismatch in keys (splits or apply_to_key) between original and transformed data.
        ValueError
            If multi-source transform changes the number of sources.
        """
        # Delegate the transformation application to the chosen strategy
        # The strategy is responsible for handling the internal structure of `data`
        # (single dict vs. list of dicts) and using `apply_to` and `fit_on`.
        transformed_data = self.strategy.apply(
            self.transform_instance, 
            data, 
            apply_to_key=self.apply_to, 
            fit_on_split=self.fit_on,
            auxiliary_keys = self.auxiliary_keys
        )
        
        # --- Post-transformation Validation ---
        # This centralizes validation logic, ensuring consistency regardless of strategy.

        if isinstance(data, dict): # Validate for single-source transformations
            if not isinstance(transformed_data, dict):
                raise TypeError(
                    f"Transform '{self.transform_name}': Strategy returned {type(transformed_data)}, "
                    f"expected dict for single source input."
                )
            
            # Ensure the top-level 'apply_to' key is present in the transformed data
            if self.apply_to not in transformed_data:
                 raise KeyError(
                     f"Transform '{self.transform_name}': Transformed data from strategy does not "
                     f"contain '{self.apply_to}' key at the top level."
                 )
            
            # Ensure the value at 'apply_to' is still a dict of splits
            if not isinstance(transformed_data[self.apply_to], dict):
                raise TypeError(
                    f"Transform '{self.transform_name}': Data at '{self.apply_to}' key in transformed data "
                    f"is of type {type(transformed_data[self.apply_to])}, expected dict of splits."
                )

            # Validate that all sub-keys (splits) within 'apply_to' match
            original_splits = set(data[self.apply_to].keys())
            transformed_splits = set(transformed_data[self.apply_to].keys())
            if original_splits != transformed_splits:
                raise KeyError(
                    f"Transform '{self.transform_name}': Split key mismatch at '{self.apply_to}'. "
                    f"Original splits: {original_splits}, Transformed splits: {transformed_splits}."
                )
            
            # Additional check for data types (optional but good practice)
            for split_name in original_splits:
                if not isinstance(data[self.apply_to][split_name], type(transformed_data[self.apply_to][split_name])):
                    logger.warning(f"Warning: Transform '{self.transform_name}' changed type of split '{split_name}' in '{self.apply_to}' from {type(data[self.apply_to][split_name])} to {type(transformed_data[self.apply_to][split_name])}.")


        elif isinstance(data, list): # Validate for multi-source transformations
            if not isinstance(transformed_data, list) or not all(isinstance(d, dict) for d in transformed_data):
                raise TypeError(
                    f"Transform '{self.transform_name}': Strategy returned {type(transformed_data)}, "
                    f"expected list of dicts for multi-source input."
                )
            
            if len(data) != len(transformed_data):
                raise ValueError(
                    f"Transform '{self.transform_name}': Multi-source transform changed the number of sources. "
                    f"Original: {len(data)}, Transformed: {len(transformed_data)}."
                )

            # Iterate through each transformed source to validate internal structure
            for i, (original_source, transformed_source) in enumerate(zip(data, transformed_data)):
                if not isinstance(transformed_source, dict):
                    raise TypeError(
                        f"Transform '{self.transform_name}': Transformed source at index {i} "
                        f"is of type {type(transformed_source)}, expected dict."
                    )

                if self.apply_to not in transformed_source:
                    raise KeyError(
                        f"Transform '{self.transform_name}': Transformed source at index {i} "
                        f"does not contain '{self.apply_to}' key."
                    )
                
                if not isinstance(transformed_source[self.apply_to], dict):
                    raise TypeError(
                        f"Transform '{self.transform_name}': Data at '{self.apply_to}' key in transformed "
                        f"source {i} is of type {type(transformed_source[self.apply_to])}, expected dict of splits."
                    )
                
                original_splits = set(original_source[self.apply_to].keys())
                transformed_splits = set(transformed_source[self.apply_to].keys())
                if original_splits != transformed_splits:
                    raise KeyError(
                        f"Transform '{self.transform_name}': Split key mismatch at '{self.apply_to}' "
                        f"for source index {i}. Original splits: {original_splits}, Transformed splits: {transformed_splits}."
                    )
                # Additional check for data types (optional but good practice)
                for split_name in original_splits:
                    if not isinstance(original_source[self.apply_to][split_name], type(transformed_source[self.apply_to][split_name])):
                        logger.warning(f"Warning: Transform '{self.transform_name}' changed type of split '{split_name}' in '{self.apply_to}' for source {i} from {type(original_source[self.apply_to][split_name])} to {type(transformed_source[self.apply_to][split_name])}.")

        else:
            raise TypeError(
                f"Transform '{self.transform_name}': Unexpected input data type: {type(data)}. "
                f"Expected dict or list of dicts."
            )

        return transformed_data