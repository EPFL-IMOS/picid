import logging
from typing import Any, Dict, List, Union
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
import omegaconf


# Import your strategy and base transform classes
from lmetk.transforms.base.base_transform import BaseTransform
from lmetk.transforms.base.strategy import TransformStrategy
from lmetk.data.data_objects.data import SplitDatasetContainer 

from collections.abc import Mapping, Sequence

def is_sequence(obj):
    """Return True for list/tuple/array-like, but False for str/bytes."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))
# To allow instantiation of Hydra-configured objects


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

        self.apply_to = metadata.get("apply_to", None)
        if self.apply_to is None:
            raise ValueError(
                f"Metadata for transform '{transform_name}' must specify 'apply_to' key."
            )
        # Ensure apply_to is a list if it is a ListConfig
        if isinstance(self.apply_to, omegaconf.listconfig.ListConfig):
            self.apply_to = list(self.apply_to)
        
        assign_to = metadata.get("assign_to", None)
        
        # Check whenever self.assign_to is None, if so, assign self.apply_to to it
        if assign_to is None:
            # Ensure that in such a case self.apply_to is string, not a list, so no ambiguity
            if is_sequence(self.apply_to):
                 raise ValueError(
                    f"Metadata for transform '{transform_name}' has 'apply_to' as a list but 'assign_to' is not specified. Please specify 'assign_to' to avoid ambiguity."
                )
            self.assign_to = self.apply_to 
            self.assign_to_map = self.apply_to 

        elif isinstance(assign_to, Mapping):   
            self.assign_to_map = list(assign_to.keys())
            self.assign_to = list(assign_to.values())
        
        elif isinstance(assign_to, str):
            self.assign_to = [assign_to]
            self.assign_to_map = [assign_to]
        
        elif is_sequence(assign_to):
            self.assign_to = assign_to
            self.assign_to_map = assign_to
        
        else: 
            raise ValueError('Unsupported type assign_to ')

        
        # TODO: is there actually a use case with fitting on validation/test?
        self.fit_on_split = metadata.get("fit_on", None)
        assert self.fit_on_split in (None, "train", "val", "test")
        # Define fit on key, if not specified, it is the same as apply_to
        self.fit_on_key = metadata.get("fit_on_key", None)
        if self.fit_on_split is not None and self.fit_on_key is None:
            # Ensure that in such a case self.apply_to is string, not a list, so no ambiguity
            if isinstance(self.apply_to, list):
                raise ValueError(
                    f"Metadata for transform '{transform_name}' has 'apply_to' as a list but 'fit_on_key' is not specified. Please specify 'fit_on_key' to avoid ambiguity."
                )
            self.fit_on_key = self.apply_to

        # TODO: Define the logic here similar to self.fit_on_split
        self.multi_source_fit = metadata.get("multi_source_fit", False)

        # Instantiate transform
        assert isinstance(transform, BaseTransform), (
            f"Transform instance for '{transform_name}' (type: {type(transform)}) "
            "must implement BaseTransform."
        )

        self.strategy = TransformStrategy()

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

    def forward(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
            If there's a mismatch in keys (splits or apply_to_keys) between original and transformed data.
        ValueError
            If multi-source transform changes the number of sources.
        """
        # Delegate the transformation application to the chosen strategy
        # The strategy is responsible for handling the internal structure of `data`
        # (single dict vs. list of dicts) and using `apply_to_keys`, `assign_to_key`, `fit_on_split`, `fit_on_key`.
        transformed_data = self.strategy.apply(
            transform_instance=self.transform_instance,
            data=data,
            apply_to_keys=self.apply_to,
            assign_to_keys=self.assign_to,
            assign_to_keys_map=self.assign_to_map,
            fit_on_split=self.fit_on_split,
            fit_on_key=self.fit_on_key,
        )

        # --- Post-transformation Validation ---
        # This centralizes validation logic, ensuring consistency regardless of strategy.

        # Validate for single-source transformations
        if not isinstance(transformed_data, SplitDatasetContainer):
            raise TypeError(
                f"Transform '{self.transform_name}': Strategy returned {type(transformed_data)}, "
                f"expected dict for single source input."
            )

        # Ensure the top-level 'apply_to' key is present in the transformed data
        # Support both single key (str) and multiple keys (list) for assign_to
        assign_to_keys = (
            [self.assign_to] if isinstance(self.assign_to, str) else self.assign_to
        )

        apply_to_keys = (
            [self.apply_to] if isinstance(self.apply_to, str) else self.apply_to
        )

        for key in assign_to_keys:
            if key not in transformed_data:
                raise KeyError(
                    f"Transform '{self.transform_name}': Transformed data from strategy does not "
                    f"contain '{key}' key at the top level."
                )

        # Ensure the value(s) at 'assign_to' are still dicts of splits
        for key in assign_to_keys:
            if not isinstance(transformed_data[key], Mapping):
                raise TypeError(
                    f"Transform '{self.transform_name}': Data at '{key}' key in transformed data "
                    f"is of type {type(transformed_data[key])}, expected dict of splits."
                )

        # Validate that all sub-keys (splits) within 'apply_to' match
        for apply_to_key in apply_to_keys:
            original_splits = set(data[apply_to_key].keys())
            transformed_splits = set(transformed_data[apply_to_key].keys())

            if original_splits != transformed_splits:
                raise KeyError(
                    f"Transform '{self.transform_name}': Split key mismatch at '{self.apply_to}'. "
                    f"Original splits: {original_splits}, Transformed splits: {transformed_splits}."
                )

            # Additional check for data types (optional but good practice)
            for split_name in original_splits:
                curr_data = data[apply_to_key][split_name]
                curr_transformed_data = transformed_data[apply_to_key][split_name]
                
                if not isinstance(curr_data, type(curr_transformed_data)):
                    logger.warning(
                        f"Warning: Transform '{self.transform_name}' changed type of split '{split_name}' in '{apply_to_key}' from {type(data[apply_to_key][split_name])} to {type(transformed_data[apply_to_key][split_name])}."
                    )
                
                #TODO: ETTH will not work if no isinstance(curr_transformed_data,list) as curr_transformed_data is ndarray but not list[ndarray]
                if isinstance(curr_transformed_data,list) & (len(curr_data) != len(curr_transformed_data)):
                    raise ValueError(
                        f"Transform '{self.transform_name}': Multi-source transform changed the number of sources. "
                        f"Original: {len(data)}, Transformed: {len(transformed_data)}."
                    )
        # TODO: additional checks,
        return transformed_data
