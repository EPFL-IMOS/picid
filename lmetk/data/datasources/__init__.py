"""DataSources with automated class discovery and exports."""

import inspect
from importlib import util
from pathlib import Path
from typing import Any

# Import base class from separate module (like your working example)
from .single_source_loader import DataSourceLoader
from .multi_source_loader import MultiSourceLoader

class ModuleExportsManager:
    """Manages automatic discovery and registration of DataSource classes."""
    
    @staticmethod
    def is_datasource_class(obj: Any) -> bool:
        """Check if an object is a valid DataSource class.
        
        Parameters
        ----------
        obj : Any
            The object to check if it's a valid datasource class.
            
        Returns
        -------
        bool
            True if the object is a valid DataSource class (non-private class
            inheriting from DataSourceLoader), False otherwise.
        """
        return (
            inspect.isclass(obj)
            and not obj.__name__.startswith("_")
            and issubclass(obj, DataSourceLoader)
            and obj != DataSourceLoader
        )
    
    @classmethod
    def discover_datasources(cls, package_path: str) -> dict[str, type]:
        """Dynamically discover all DataSource classes in the package.
        
        Parameters
        ----------
        package_path : str
            Path to the package's __init__.py file.
            
        Returns
        -------
        dict[str, type]
            Dictionary mapping class names to their corresponding class objects.
        """
        datasources = {}
        
        # Get the directory containing the datasource modules
        package_dir = Path(package_path).parent
        
        # Iterate through all .py files in the directory
        for file_path in package_dir.glob("*.py"):
            if file_path.stem in ["__init__", "base"]:  # Skip __init__ and base
                continue
                
            # Import the module
            module_name = f"{Path(package_path).stem}.{file_path.stem}"
            spec = util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all datasource classes in the module
                new_datasources = {
                    name: obj
                    for name, obj in inspect.getmembers(module)
                    if (
                        inspect.isclass(obj)
                        and obj.__module__ == module.__name__
                        and not name.startswith("_")
                        and issubclass(obj, DataSourceLoader)
                        and obj != DataSourceLoader
                    )
                }
                datasources.update(new_datasources)
                
        return datasources


# Create the exports manager
manager = ModuleExportsManager()

# Automatically discover and populate ALL_CLASSES
ALL_CLASSES = manager.discover_datasources(__file__)

# Automatically generate __all__
__all__ = [
    *ALL_CLASSES.keys(),
    "DataSourceLoader", 
    "ALL_CLASSES",
]

# For backwards compatibility, create individual imports in local namespace
locals().update(**ALL_CLASSES)