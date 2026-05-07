# Docstring Style Guide

PICID uses **NumPy-style** docstrings for API documentation. mkdocstrings is
configured with `docstring_style: numpy`; other styles (e.g. Google) will not
render Parameters/Returns correctly.

## Required format

- **Parameters:** Use `Parameters` + `----------`; each param as `name : type` + indented description.
- **Returns:** Use `Returns` + `-------`; type and description on separate lines.
- **Raises:** Use `Raises` + `------`.
- **Examples:** Use `Examples` + `--------`.

See [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
