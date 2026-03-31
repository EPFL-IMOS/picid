"""Splitter that partitions whole datasource payloads by source identity."""

from typing import Any, Dict, List, Tuple

import logging

from picid.data.data_objects import DatasetContainer

logger = logging.getLogger(__name__)


class BySourceSplitter:
    """
    Assign whole datasource payloads to train/val/test splits.

    Parameters
    ----------
    sources_train : list[str] | None, optional
        Source names assigned to the training split.
    sources_val : list[str] | None, optional
        Source names assigned to the validation split.
    sources_test : list[str] | None, optional
        Source names assigned to the test split.
    """

    def __init__(
        self,
        sources_train: List[str] = None,
        sources_val: List[str] = None,
        sources_test: List[str] = None,
    ):
        """
        Initialize the source-level split assignment.

        Parameters
        ----------
        sources_train : list[str] | None, optional
            Source names assigned to the training split.
        sources_val : list[str] | None, optional
            Source names assigned to the validation split.
        sources_test : list[str] | None, optional
            Source names assigned to the test split.
        """
        self.sources_train = sources_train if sources_train is not None else []
        self.sources_val = sources_val if sources_val is not None else []
        self.sources_test = sources_test if sources_test is not None else []

        # Validate that each source belongs to exactly one split so the
        # multisource parity and cache assumptions remain deterministic.
        all_sources = (
            set(self.sources_train) | set(self.sources_val) | set(self.sources_test)
        )
        if len(all_sources) != (
            len(self.sources_train) + len(self.sources_val) + len(self.sources_test)
        ):
            raise ValueError(
                "Sources must be disjoint sets. Some sources appear in multiple splits."
            )

    def split_data(
        self, data_list: List[DatasetContainer], source_name_lst: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Split source payloads into a split-aware dataset mapping.

        Parameters
        ----------
        data_list : list[DatasetContainer]
            Loaded datasource payloads in the same order as ``source_name_lst``.
        source_name_lst : list[str]
            Names corresponding to each payload in ``data_list``.

        Returns
        -------
        dict[str, dict[str, Any]]
            Split-aware mapping keyed first by payload name and then by split.
        """

        if len(data_list) != len(source_name_lst):
            raise ValueError(
                f"Length of data_list={len(data_list)} and source_name_lst={len(source_name_lst)} must be equal."
            )

        sources = dict(zip(source_name_lst, data_list))

        split_data = {
            key: {"train": None, "val": None, "test": None}
            for key in data_list[0].keys()
        }

        for split, split_sources_list in zip(
            ["train", "val", "test"],
            [self.sources_train, self.sources_val, self.sources_test],
        ):
            # Gather the whole payload for each source in the configured split.
            selected_data_dicts = [sources[key] for key in split_sources_list]

            # Reorganize the list of source payloads into per-key lists.
            stacked = convert_outer_list_to_inner(selected_data_dicts)

            # Attach each per-key payload list to the correct split branch.
            for key in stacked:
                split_data[key][split] = stacked[key]

        logger.info("Split tree view:")
        for split, split_sources_list in zip(
            ["train", "val", "test"],
            [self.sources_train, self.sources_val, self.sources_test],
        ):
            logger.info(f"|- {split}")
            for source in split_sources_list:
                logger.info(f"   |- {source}")

        return split_data

    def __call__(
        self, data_list: List[Dict[str, Any]], source_name_lst: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], str]:
        """
        Split data and return the corresponding human-readable tree view.

        Parameters
        ----------
        data_list : list[dict[str, Any]]
            Source payloads to partition.
        source_name_lst : list[str]
            Source names aligned with ``data_list``.

        Returns
        -------
        tuple[dict[str, dict[str, Any]], str]
            The split payload mapping and the textual split tree view.
        """
        sources_by_split = {
            "train": self.sources_train,
            "val": self.sources_val,
            "test": self.sources_test,
        }

        return self.split_data(data_list, source_name_lst), self.get_split_tree_view(
            sources_by_split
        )

    def get_split_tree_view(self, sources_by_split: dict[str, list[str]]) -> str:
        """
        Render the source assignment tree for logging and debugging.

        Parameters
        ----------
        sources_by_split : dict[str, list[str]]
            Mapping from split name to the assigned source names.

        Returns
        -------
        str
            Multi-line textual tree representation of the split layout.
        """
        lines = ["Split tree view:"]
        for split in ["train", "val", "test"]:
            lines.append(f"|- {split}")
            for source in sources_by_split.get(split, []):
                lines.append(f"   |- {source}")
        return "\n".join(lines)


def convert_outer_list_to_inner(
    data_list: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """
    Convert a list of keyed payloads into a keyed list-of-payloads mapping.

    Parameters
    ----------
    data_list : list[dict[str, Any]]
        Source payloads with matching keys.

    Returns
    -------
    dict[str, list[Any]]
        Mapping from payload key to the list of per-source values.
    """
    if not data_list:
        return {}

    # Ensure all dictionaries have the same keys
    keys = data_list[0].keys()
    for d in data_list:
        if d.keys() != keys:
            raise ValueError("Not all dicts have the same keys!")

    # Concatenate values for each key
    stacked = {}
    for key in keys:
        stacked[key] = [d[key] for d in data_list]

    # Check that all keys in the stacked have the same length
    lengths = [len(stacked[key]) for key in stacked]
    if len(set(lengths)) != 1:
        logger.error(
            f"Not all keys in the stacked dict have the same length. Lengths: {lengths}"
        )
        raise ValueError("All keys in the stacked dict must have the same length.")

    return stacked
