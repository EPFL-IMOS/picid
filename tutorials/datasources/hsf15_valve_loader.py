#!/usr/bin/env python3
"""Tutorial: HSF15 valve subsystem via HSF15Loader."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutorials.datasources.hsf15_component_loader import (
    main_for_component,
    parse_wrapper_args,
)


def main() -> None:
    args = parse_wrapper_args()
    main_for_component("valve", data_dir_cli=args.data_dir)


if __name__ == "__main__":
    main()
