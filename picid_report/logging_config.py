"""
Central logging configuration for the picid_report pipeline.

Use configure_logging(debug=...) at pipeline entry (e.g. run.main()) to set:
- INFO: stage boundaries, per-dataset/model decisions, counts (suitable for "stay tuned").
- DEBUG: inner details (resolver in/out, column lists, shapes) for debugging.

When debug=False (default), only INFO and above are shown with a simple message format.
When debug=True, DEBUG and above are shown with level and module name for tracing.
"""

import logging
import sys

# Root logger name for the package; all subloggers (picid_report.run, etc.) inherit.
LOGGER_NAME = "picid_report"

# Format when debug is off: just the message (current behavior).
FORMAT_INFO = "%(message)s"
# Format when debug is on: level and module for tracing.
FORMAT_DEBUG = "[%(levelname)s] %(name)s: %(message)s"


def configure_logging(
    debug: bool = False,
    stream=None,
) -> None:
    """
    Configure the picid_report package logger and its children.

    Call this at pipeline entry (e.g. in run.main()) so that:
    - debug=False: INFO and above, message-only format (clean for normal runs).
    - debug=True: DEBUG and above, level+name+message (for debugging).

    Parameters
    ----------
    debug : bool
        If True, set level to DEBUG and use verbose format; otherwise INFO and simple format.
    stream : file-like, optional
        Where to emit logs; default is sys.stdout.
    """
    if stream is None:
        stream = sys.stdout
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    # Avoid duplicate handlers when main() is called multiple times (e.g. tests).
    log.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter(FORMAT_DEBUG if debug else FORMAT_INFO))
    log.addHandler(handler)
    # Prevent propagation to root so we control output in one place.
    log.propagate = False
