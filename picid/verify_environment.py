import sys
import importlib.util
import pkgutil
import importlib
import lmetk

print("=== sys.path ===")
for p in sys.path:
    print(p)

print("\n=== Trying import lmetk ===")
spec = importlib.util.find_spec("lmetk")
if spec is None:
    print("lmetk NOT found")
else:
    print(f"lmetk found at: {spec.origin}")

    import lmetk

    print("lmetk module object:", lmetk)
    print("lmetk __file__:", getattr(lmetk, "__file__", None))
    print("lmetk __path__:", getattr(lmetk, "__path__", None))

print("\n=== Installed distributions matching lmetk ===")
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata  # backport for <3.8

for dist in metadata.distributions():
    if "lmetk" in dist.metadata["Name"].lower():
        print("Distribution:", dist.metadata["Name"], dist.version)


def walk_packages(package):
    for _, name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        print(name, "(pkg)" if ispkg else "")
        if ispkg:
            mod = importlib.import_module(name)
            walk_packages(mod)


walk_packages(lmetk)
