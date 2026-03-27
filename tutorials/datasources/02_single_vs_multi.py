#!/usr/bin/env python3
"""Tutorial: Single vs multi-source split modes (per_unit vs cross_unit)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.data.datasources.toy_example import ToyRaggedLoader
from picid.data.data_objects import SplitDatasetContainer


class MultiSourceToyLoader(ToyRaggedLoader):
    """Toy loader with multisource_data_splitter set to demonstrate cross_unit mode.

    Extends ToyRaggedLoader and passes multisource_data_splitter=object() so that
    get_split_mode() returns "cross_unit". We reuse ToyRaggedLoader's _load_data
    logic; no actual PHMD multi-source data is needed for this demo.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("multisource_data_splitter", object())
        super().__init__(**kwargs)


def main():
    # --- Single source: ToyRaggedLoader (no multisource_data_splitter) ---
    single = ToyRaggedLoader(
        data_dir=".", data_name="toy", task_mode="anomaly_detection"
    )
    single.load_data()
    assert single.get_split_mode() == "per_unit"
    container_single = single.get_data()
    assert isinstance(container_single, SplitDatasetContainer)
    assert "train" in container_single.features
    assert "test" in container_single.features

    # --- Multi source: MultiSourceToyLoader (multisource_data_splitter=object()) ---
    multi = MultiSourceToyLoader(
        data_dir=".", data_name="toy_multi", task_mode="anomaly_detection"
    )
    multi.load_data()
    assert multi.get_split_mode() == "cross_unit"
    container_multi = multi.get_data()
    assert isinstance(container_multi, SplitDatasetContainer)
    assert "train" in container_multi.features
    assert "test" in container_multi.features

    # --- Compare and print ---
    print(f"Single (ToyRaggedLoader):        split_mode = {single.get_split_mode()}")
    print(f"Multi  (MultiSourceToyLoader):   split_mode = {multi.get_split_mode()}")
    print("OK")


if __name__ == "__main__":
    main()
