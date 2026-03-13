# Cache Keys and File Locking

## Cache key inputs

Cache identity is derived from:

- datasource init config
- transform config
- code fingerprint from relevant library paths

Any change invalidates cache reuse.

## Concurrency safety

Optional file locking in `run.py` prevents concurrent preprocessors from writing the same cache directory simultaneously.

This is required in shared compute environments.
