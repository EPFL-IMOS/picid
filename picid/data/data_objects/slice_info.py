"""
Slice-awareness — coordinate selections and bounds attached to the container.

Enables downstream transforms to query "what slice am I operating on?" and
"what coordinate system is this input under?" without changing existing APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import copy as copy_mod


@dataclass
class SliceInfo:
    """
    Describes the coordinate slice / selection for the current data.

    Optional fields allow datasource or preprocessing to record which units,
    cycles, or time range the container represents. Transforms can read this
    from the container or from TransformContext.slice_info, or from the
    metadata dict passed to transform_data/fit_data (under the key "slice_info",
    as a dict), when the transform opts in via metadata.include_slice_info_in_metadata: true.

    Attributes
    ----------
    split : Optional[str]
        Split name (e.g. "train", "val", "test") when the slice is split-scoped.
    unit_ids : Optional[List[Any]]
        Unit identifiers included in this slice (e.g. unit indices or IDs).
    cycle_ids : Optional[List[Any]]
        Cycle identifiers when applicable.
    bounds : Optional[Dict[str, Any]]
        Optional bounds (e.g. time_min, time_max, or other coordinate bounds).
    index_map : Optional[Dict[str, Any]]
        Placeholder for original→sliced index mapping when applicable.
    """

    split: Optional[str] = None
    unit_ids: Optional[List[Any]] = None
    cycle_ids: Optional[List[Any]] = None
    bounds: Optional[Dict[str, Any]] = None
    index_map: Optional[Dict[str, Any]] = None

    def copy(self, deep: bool = True) -> "SliceInfo":
        """
        Return a copy of the slice description.

        Parameters
        ----------
        deep : bool, default=True
            Whether nested list and dictionary fields should be deep-copied.

        Returns
        -------
        SliceInfo
            Copied slice description.
        """
        if deep:
            return SliceInfo(
                split=self.split,
                unit_ids=copy_mod.deepcopy(self.unit_ids) if self.unit_ids else None,
                cycle_ids=copy_mod.deepcopy(self.cycle_ids) if self.cycle_ids else None,
                bounds=copy_mod.deepcopy(self.bounds) if self.bounds else None,
                index_map=copy_mod.deepcopy(self.index_map) if self.index_map else None,
            )
        return SliceInfo(
            split=self.split,
            unit_ids=list(self.unit_ids) if self.unit_ids else None,
            cycle_ids=list(self.cycle_ids) if self.cycle_ids else None,
            bounds=dict(self.bounds) if self.bounds else None,
            index_map=dict(self.index_map) if self.index_map else None,
        )
