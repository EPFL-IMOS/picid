"""Module with automated class discovery and exports."""

import inspect
from importlib import util
from pathlib import Path
from typing import ClassVar, Optional, Type
from torch.utils.data import Dataset


class ClassManager:
    """Manages automatic discovery and registration of classes."""

    @classmethod
    def discover_classes(
        cls, package_path: str, base_class: Optional[Type] = None
    ) -> dict[str, Type]:
        """Dynamically discover classes in the package.

        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.
        base_class : Type, optional
            If provided, only classes that inherit from this base class will be included.

        Returns
        -------
        Dict[str, Type]
            Dictionary mapping class names to their corresponding class objects.
        """
        classes = {}

        # Get the directory containing the modules
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

                # Find all classes in the module matching criteria
                new_classes = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                        and (base_class is None or issubclass(obj, base_class))
                    )
                }
                classes.update(new_classes)
        return classes


# Create the class manager
manager = ClassManager()

# Automatically discover all classes
ALL_CLASSES = manager.discover_classes(__file__)

# Automatically generate __all__
__all__ = [
    # Class collections
    "ALL_CLASSES",
    # Discovered class names
    *ALL_CLASSES.keys(),
]

# Create individual imports in the local namespace
locals().update(**ALL_CLASSES)
