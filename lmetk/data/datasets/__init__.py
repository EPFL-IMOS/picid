"""Dataset module with automated exports."""

import inspect
from importlib import util
from pathlib import Path
from typing import ClassVar
from torch.utils.data import Dataset

class DatasetManager:
    """Manages automatic discovery and registration of dataset classes."""

    # Static dataset definitions
    

    @classmethod
    def discover_datasets(
        cls, package_path: str
    ) -> dict[str, Dataset]:
        """Dynamically discover all dataset classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.

        Returns
        -------
        Dict[str, Dataset]
            Dictionary mapping class names to their corresponding class objects.
        """
        datasets = {}

        # Get the directory containing the dataset modules
        package_dir = Path(package_path).parent

        # Iterate through all .py files in the directory
        for file_path in package_dir.glob("*.py"):
            if file_path.stem == "__init__":
                continue

            # Import the module
            module_name = f"{Path(package_path).stem}.{file_path.stem}"
            spec = util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all dataset classes in the module
                new_datasets = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                        and issubclass(obj, Dataset)
                        and obj != Dataset
                    )
                }
                datasets.update(new_datasets)
        return datasets



# Create the dataset manager
manager = DatasetManager()

# Automatically discover and populate datasets
MANUAL_DATASETS = manager.discover_datasets(__file__)

# Automatically generate __all__
__all__ = [
    # Dataset collections
    "MANUAL_DATASETS",
    # Discovered dataset classes
    *MANUAL_DATASETS.keys(),
]

# For backwards compatibility, create individual imports
locals().update(**MANUAL_DATASETS)