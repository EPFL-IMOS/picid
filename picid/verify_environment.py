import sys
import importlib.util
import pkgutil
import importlib
import picid

print("=== sys.path ===")
for p in sys.path:
    print(p)

print("\n=== Trying import picid ===")
spec = importlib.util.find_spec("picid")
if spec is None:
    print("picid NOT found")
else:
    print(f"picid found at: {spec.origin}")

    import picid

    print("picid module object:", picid)
    print("picid __file__:", getattr(picid, "__file__", None))
    print("picid __path__:", getattr(picid, "__path__", None))

print("\n=== Installed distributions matching picid ===")
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata  # backport for <3.8

for dist in metadata.distributions():
    if "picid" in dist.metadata["Name"].lower():
        print("Distribution:", dist.metadata["Name"], dist.version)


def walk_packages(package):
    for _, name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        print(name, "(pkg)" if ispkg else "")
        if ispkg:
            mod = importlib.import_module(name)
            walk_packages(mod)


walk_packages(picid)
