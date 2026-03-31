"""
Configuration-driven transform manager.

ConfigTransformManager builds DataTransform instances from a config (e.g. Hydra),
indexes them by name, and provides forward/inverse application over a
SplitDatasetContainer. Used by the run pipeline and by dry_run to resolve
transform lists without executing heavy compute.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from picid.transforms.base.data_transform import DataTransform
from picid.utils.hash_utils import ensure_serializable
from picid.transforms.base.multisource import InverseTransformMixin

logger = logging.getLogger(__name__)


class ConfigTransformManager:
    """
    Manage transform configs and instantiated `DataTransform` objects.

    Parameters
    ----------
    transforms_config : dict or DictConfig, optional
        Mapping of transform names to transform configuration entries.
    lazy_instantiation : bool, default=False
        Whether to delay Hydra instantiation until the transforms are first
        requested.
    """

    def __init__(
        self,
        transforms_config: Optional[Union[Dict, DictConfig]] = None,
        lazy_instantiation: bool = False,
    ):
        """
        Initialize the transform manager from configuration.

        Parameters
        ----------
        transforms_config : dict or DictConfig, optional
            Transforms configuration dictionary or DictConfig.
        lazy_instantiation : bool
            If False (default), instantiate transforms immediately.
        """
        self.config = self._ensure_dict_config(transforms_config or {})
        self.lazy_instantiation = lazy_instantiation

        # Core storage - OrderedDict for efficient access to DataTransform objects
        self._data_transforms: Optional[OrderedDict[str, DataTransform]] = None
        self._is_instantiated = False

        # Instantiate immediately unless lazy mode
        if not lazy_instantiation and self.config:
            self._instantiate_all()

    def _ensure_dict_config(self, config: Union[Dict, DictConfig]) -> DictConfig:
        """
        Convert plain dictionaries to `DictConfig`.

        Parameters
        ----------
        config : dict or DictConfig
            Input configuration to normalize.

        Returns
        -------
        DictConfig
            Normalized OmegaConf dictionary configuration.
        """
        if isinstance(config, dict):
            return OmegaConf.create(config)
        return config

    def _instantiate_all(self) -> None:
        """Instantiate all configured `DataTransform` objects."""
        if self._is_instantiated:
            return

        data_transforms = OrderedDict()

        for transform_name, transform_entry_config in self.config.items():
            # transform_name start from the __ then skip # TODO: why?
            if transform_name.startswith("__"):
                continue

            # Get the transform configuration
            transform_config = transform_entry_config.get("transform")

            if transform_config is None:
                logger.error(
                    f"Transform '{transform_name}' missing 'transform' configuration"
                )

            # Instantiate the raw transform using Hydra
            instantiated_transform = instantiate(transform_config)

            # assert the instantiated_transform is instantiated:
            if instantiated_transform is None:
                logger.error(
                    f"Failed to instantiate transform '{transform_name}' from config: {transform_config}"
                )
                raise RuntimeError(
                    f"Failed to instantiate transform '{transform_name}' from config: {transform_config}"
                )

            # Get metadata
            metadata = transform_entry_config.get("metadata", {})

            # Create DataTransform object
            data_transform = DataTransform(
                transform_name=transform_name,
                transform=instantiated_transform,
                metadata=metadata,
            )

            data_transforms[transform_name] = data_transform

        self._data_transforms = data_transforms
        self._is_instantiated = True

    def add_transforms_config(self, transforms_config: Union[Dict, DictConfig]) -> None:
        """
        Add transforms from configuration.

        Parameters
        ----------
        transforms_config : dict or DictConfig
            Configuration dictionary with transform definitions.

        Raises
        ------
        RuntimeError
            If transforms are already instantiated (must clear cache first)
        KeyError
            If transform name already exists
        """
        if self._is_instantiated:
            raise RuntimeError(
                "Cannot add transforms after instantiation. "
                "Call clear_cache() first to reset the manager state."
            )

        new_config = self._ensure_dict_config(transforms_config)

        # Check for conflicts
        conflicts = [name for name in new_config.keys() if name in self.config]
        if conflicts:
            raise KeyError(f"Transform names already exist: {conflicts}")

        # Merge with existing config
        self.config = OmegaConf.merge(self.config, new_config)

        # Auto-instantiate if not in lazy mode
        if not self.lazy_instantiation:
            self._instantiate_all()

    def update_transforms_config(
        self, transforms_config: Union[Dict, DictConfig]
    ) -> None:
        """
        Update/replace transforms configuration.

        Parameters
        ----------
        transforms_config : dict or DictConfig
            New transforms configuration.

        Raises
        ------
        RuntimeError
            If transforms are already instantiated (must clear cache first)
        """
        if self._is_instantiated:
            raise RuntimeError(
                "Cannot update transforms after instantiation. "
                "Call clear_cache() first to reset the manager state."
            )

        self.config = self._ensure_dict_config(transforms_config)

        # Auto-instantiate if not in lazy mode
        if not self.lazy_instantiation and self.config:
            self._instantiate_all()

    def has_transform(self, name: str) -> bool:
        """
        Check whether a named transform exists.

        Parameters
        ----------
        name : str
            Transform name to look up.

        Returns
        -------
        bool
            Whether the transform exists in the stored configuration.
        """
        return name in self.config

    def get_transform_names(self) -> List[str]:
        """
        Return all transform names in configuration order.

        Returns
        -------
        list of str
            Transform names in their configured pipeline order.
        """
        return list(self.config.keys())

    def get_transforms_config(self) -> DictConfig:
        """
        Return the stored transforms configuration.

        Returns
        -------
        DictConfig
            Full transform configuration.
        """
        return self.config

    def get_transform_config(self, name: str) -> DictConfig:
        """
        Get configuration for a specific transform.

        Parameters
        ----------
        name : str
            Transform name.

        Returns
        -------
        DictConfig
            Transform configuration.
        """
        if name not in self.config:
            raise KeyError(f"Transform '{name}' not found")

        return self.config[name]

    def get_data_transforms(self) -> OrderedDict[str, DataTransform]:
        """
        Get all instantiated DataTransform objects as OrderedDict.
        This is the primary method for accessing DataTransform objects.

        Returns
        -------
        OrderedDict
            OrderedDict with all instantiated DataTransform objects
        """
        if not self._is_instantiated:
            self._instantiate_all()

        return self._data_transforms.copy() if self._data_transforms else OrderedDict()

    def get_data_transform(self, name: str) -> DataTransform:
        """
        Return one instantiated `DataTransform` by name.

        Parameters
        ----------
        name : str
            Transform name.

        Returns
        -------
        DataTransform
            Instantiated transform wrapper.
        """
        if not self._is_instantiated:
            self._instantiate_all()

        if not self._data_transforms or name not in self._data_transforms:
            raise KeyError(f"Transform '{name}' not found")

        return self._data_transforms[name]

    def prepare_transforms(self) -> List[DataTransform]:
        """
        Return instantiated transforms as a list.

        Returns
        -------
        list of DataTransform
            Instantiated transforms in pipeline order.
        """
        data_transforms = self.get_data_transforms()
        return list(data_transforms.values())

    def get_transforms_by_apply_to(self, apply_to: str) -> List[DataTransform]:
        """
        Return transforms that write to a specific data key.

        Parameters
        ----------
        apply_to : str
            Data type (for example, ``features`` or ``target``).

        Returns
        -------
        OrderedDict
            DataTransform objects filtered by ``apply_to``.
        """
        data_transforms = self.get_data_transforms()
        return OrderedDict(
            (dt.transform_name, dt)
            for dt in data_transforms.values()
            if dt.metadata.get("apply_to") == apply_to
        )

    def get_transforms_by_fit_on(self, fit_on: str) -> List[DataTransform]:
        """
        Return transforms that should be fit on a specific split.

        Parameters
        ----------
        fit_on : str
            Split name (for example, ``train`` or ``val``).

        Returns
        -------
        OrderedDict
            DataTransform objects filtered by ``fit_on``.
        """
        data_transforms = self.get_data_transforms()
        return OrderedDict(
            (dt.transform_name, dt)
            for dt in data_transforms.values()
            if dt.metadata.get("fit_on") == fit_on
        )

    def _pipeline_order(self) -> List[str]:
        """
        Return transform names in pipeline order.

        Returns
        -------
        list of str
            Configured transform names excluding private ``__`` keys.
        """
        return [n for n in self.config.keys() if not str(n).startswith("__")]

    def get_cache_point_names(self) -> List[str]:
        """
        Return transform names marked as boundary cache points.

        Returns
        -------
        list of str
            Transform names marked with ``metadata.cache_point = True``.
        """
        order = self._pipeline_order()
        cache_points = []
        for name in order:
            entry = self.config.get(name)
            if entry is None:
                continue
            meta = entry.get("metadata") or {}
            if meta.get("cache_point", False):
                cache_points.append(name)
        return cache_points

    def get_transform_names_after(self, transform_name: str) -> List[str]:
        """
        Return transform names that follow a named transform.

        Parameters
        ----------
        transform_name : str
            Name of the transform we have already applied (boundary).

        Returns
        -------
        list of str
            Transform names to run after this boundary.
        """
        order = self._pipeline_order()
        try:
            idx = order.index(transform_name)
            return order[idx + 1 :]
        except ValueError:
            return list(order)

    def get_config_up_to_and_including(self, transform_name: str) -> Dict:
        """
        Return a serializable config slice up to a named transform.

        Parameters
        ----------
        transform_name : str
            Last transform name to include.

        Returns
        -------
        dict
            Plain dict suitable for hashing (e.g. compute_cache_key).
        """
        order = self._pipeline_order()
        try:
            idx = order.index(transform_name)
        except ValueError:
            idx = len(order) - 1
        subset = {k: self.config[k] for k in order[: idx + 1]}
        return ensure_serializable(subset)

    def remove_transform(self, name: str) -> None:
        """
        Remove a transform from configuration.

        Parameters
        ----------
        name : str
            Transform name to remove.

        Raises
        ------
        RuntimeError
            If transforms are already instantiated (must clear cache first)
        """
        if self._is_instantiated:
            raise RuntimeError(
                "Cannot remove transforms after instantiation. "
                "Call clear_cache() first to reset the manager state."
            )

        if name not in self.config:
            raise KeyError(f"Transform '{name}' not found")

        del self.config[name]

    def clear_cache(self) -> None:
        """
        Clear instantiated DataTransform objects and reset manager state.
        After calling this, you can modify configuration again.
        """
        self._data_transforms = None
        self._is_instantiated = False

    def force_reinstantiate(self) -> None:
        """
        Force re-instantiation of all transforms from current configuration.
        Useful when you want to reload transforms without clearing config.
        """
        self.clear_cache()
        if self.config:
            self._instantiate_all()

    @property
    def is_instantiated(self) -> bool:
        """
        Check whether transforms have already been instantiated.

        Returns
        -------
        bool
            Whether the manager currently has instantiated transforms cached.
        """
        return self._is_instantiated

    # Backward compatibility methods
    def instantiate_transforms(self) -> OrderedDict[str, DataTransform]:
        """
        Return all instantiated transforms.

        Returns
        -------
        OrderedDict
            Alias for :meth:`get_data_transforms`.
        """
        return self.get_data_transforms()

    def get_transforms(self) -> OrderedDict[str, DataTransform]:
        """
        Return all instantiated transforms.

        Returns
        -------
        OrderedDict
            Alias for :meth:`get_data_transforms`.
        """
        return self.get_data_transforms()

    def get_transform(self, name: str) -> DataTransform:
        """
        Return one instantiated transform by name.

        Parameters
        ----------
        name : str
            Transform name.

        Returns
        -------
        DataTransform
            Alias for :meth:`get_data_transform`.
        """
        return self.get_data_transform(name)

    def get_inverter_for_key(
        self, key: str, which: str = "last"
    ) -> Optional[InverseTransformMixin]:
        """
        Return an inverse-capable transform for a given output key.

        When several transforms assign to the same key and implement inverse,
        pipeline order is used: ``"last"`` returns the last writer and
        ``"first"`` returns the first writer.

        Parameters
        ----------
        key : str
            Data key the transform assigns to (e.g. "target", "features").
        which : str
            ``"last"`` (default) or ``"first"`` in pipeline order.

        Returns
        -------
        InverseTransformMixin or None
            Matching inverse transform instance, or ``None`` if no match.
        """
        if which not in ("first", "last"):
            raise ValueError(
                f"get_inverter_for_key(which=...) must be 'first' or 'last', got {which!r}"
            )
        if not self._is_instantiated:
            self._instantiate_all()
        if not self._data_transforms:
            return None
        inverter = None
        for dt in self._data_transforms.values():
            assign_to = getattr(dt, "assign_to", None)
            if assign_to is None:
                continue
            keys = [assign_to] if isinstance(assign_to, str) else list(assign_to)
            if key not in keys:
                continue
            if isinstance(dt.transform_instance, InverseTransformMixin):
                inverter = dt.transform_instance
                if which == "first":
                    return inverter
        return inverter

    def get_inverter_for_key_with_name(
        self, key: str, which: str = "last"
    ) -> Tuple[Optional[InverseTransformMixin], Optional[str]]:
        """
        Return an inverse-capable transform and its config name.

        Parameters
        ----------
        key : str
            Data key the transform assigns to.
        which : str
            ``"last"`` (default) or ``"first"`` in pipeline order.

        Returns
        -------
        tuple
            ``(transform_instance, transform_name)``, or ``(None, None)`` if
            no match exists.
        """
        if which not in ("first", "last"):
            raise ValueError(
                f"get_inverter_for_key_with_name(which=...) must be 'first' or 'last', got {which!r}"
            )
        if not self._is_instantiated:
            self._instantiate_all()
        if not self._data_transforms:
            return None, None
        inverter = None
        inverter_name = None
        for dt in self._data_transforms.values():
            assign_to = getattr(dt, "assign_to", None)
            if assign_to is None:
                continue
            keys = [assign_to] if isinstance(assign_to, str) else list(assign_to)
            if key not in keys:
                continue
            if isinstance(dt.transform_instance, InverseTransformMixin):
                inverter = dt.transform_instance
                inverter_name = dt.transform_name
                if which == "first":
                    return inverter, inverter_name
        return inverter, inverter_name

    def __len__(self) -> int:
        """
        Return the number of configured transforms.

        Returns
        -------
        int
            Number of configured transforms.
        """
        return len(self.config)

    def __contains__(self, name: str) -> bool:
        """
        Check whether a transform exists.

        Parameters
        ----------
        name : str
            Transform name.

        Returns
        -------
        bool
            Whether the transform exists.
        """
        return self.has_transform(name)

    def __repr__(self) -> str:
        """String representation."""
        status = "instantiated" if self._is_instantiated else "not instantiated"
        return f"ConfigTransformManager(transforms={len(self.config)}, {status})"


# Factory function for easy creation
def create_transform_manager_from_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    lazy_instantiation: bool = False,
) -> ConfigTransformManager:
    """
    Create transform manager from configuration file or dictionary.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file.
    config_dict : dict, optional
        Configuration dictionary.
    lazy_instantiation : bool
        If False (default), instantiate transforms immediately.

    Returns
    -------
    ConfigTransformManager
        ConfigTransformManager instance.
    """
    if config_path:
        config = OmegaConf.load(config_path)
    elif config_dict:
        config = OmegaConf.create(config_dict)
    else:
        config = OmegaConf.create({})

    return ConfigTransformManager(
        transforms_config=config, lazy_instantiation=lazy_instantiation
    )
