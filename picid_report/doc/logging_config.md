# logging_config.py — Logging configuration

**Navigation:** [← Documentation index](README.md)

This document describes **configure_logging(debug=...)** and CLI **--debug**.

---

## 1. configure_logging(debug=False, stream=None)

Configures the picid_report package logger. Call at pipeline entry (e.g. run.main()).

- **debug=False:** INFO and above, message-only format.
- **debug=True:** DEBUG and above, format with level and module name.

**Example:**
```python
from picid_report.logging_config import configure_logging

configure_logging(debug=False)   # Normal run
configure_logging(debug=True)    # Verbose (resolver, shapes, etc.)
```

**Example - log to file:**
```python
with open("pipeline.log", "w") as f:
    configure_logging(debug=True, stream=f)
```

---

## 2. CLI --debug

```bash
python -m picid_report.run --debug -o report_output
```
Same as configure_logging(debug=True) then run_pipeline(...).

---

## 3. Constants

LOGGER_NAME = "picid_report", FORMAT_INFO (message only), FORMAT_DEBUG (level + name + message). Use configure_logging(debug=...) at entry; see [run.md](run.md) for CLI.

---

**Navigation:** [← Documentation index](README.md)
