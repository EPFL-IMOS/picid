"""
Metadata manifest — versioned, queryable metadata attached to the container.

Schema: each entry has schema_version, producer_version, category (datasource | transform),
optional scope (step_id, key, split), and a payload dict. Queryable by step_id, key, split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class ManifestEntry:
    """
    A single manifest entry: versioned, categorized, and optionally scoped.

    Attributes
    ----------
    schema_version : str
        Manifest schema version (e.g. "1.0").
    producer_version : str
        Producer/library version that wrote this entry (e.g. "0.1.0").
    category : str
        "datasource" (upstream) or "transform" (fit stats, feature definitions, etc.).
    payload : Dict[str, Any]
        Arbitrary payload (fit stats, validation outcome, etc.).
    step_id : Optional[str]
        Step or transform name that produced this entry.
    key : Optional[str]
        Data key this entry applies to (e.g. "features", "target").
    split : Optional[str]
        Split this entry applies to (e.g. "train", "val", "test").
    """

    schema_version: str
    producer_version: str
    category: str  # "datasource" | "transform"
    payload: Dict[str, Any]
    step_id: Optional[str] = None
    key: Optional[str] = None
    split: Optional[str] = None

    def __post_init__(self) -> None:
        if self.category not in ("datasource", "transform"):
            raise ValueError(
                f"category must be 'datasource' or 'transform', got {self.category!r}"
            )


class MetadataManifest:
    """
    Store manifest entries and support simple filtering by scope.

    Parameters
    ----------
    entries : list[ManifestEntry] | None, optional
        Optional initial manifest entries.
    """

    def __init__(self, entries: Optional[List[ManifestEntry]] = None) -> None:
        self._entries: List[ManifestEntry] = list(entries) if entries else []

    def add(self, entry: ManifestEntry) -> None:
        """
        Append one manifest entry.

        Parameters
        ----------
        entry : ManifestEntry
            Entry to append.
        """
        self._entries.append(entry)

    def query(
        self,
        *,
        step_id: Optional[str] = None,
        key: Optional[str] = None,
        split: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[ManifestEntry]:
        """
        Return entries matching the given filters.

        Parameters
        ----------
        step_id : str | None, optional
            Step identifier to filter by.
        key : str | None, optional
            Data key to filter by.
        split : str | None, optional
            Split name to filter by.
        category : str | None, optional
            Manifest category to filter by.

        Returns
        -------
        list[ManifestEntry]
            Entries matching the requested filters.
        """
        out = []
        for e in self._entries:
            if step_id is not None and e.step_id != step_id:
                continue
            if key is not None and e.key != key:
                continue
            if split is not None and e.split != split:
                continue
            if category is not None and e.category != category:
                continue
            out.append(e)
        return out

    def copy(self, deep: bool = True) -> "MetadataManifest":
        """
        Return a copy of the manifest.

        Parameters
        ----------
        deep : bool, default=True
            Whether payload dictionaries should be deep-copied.

        Returns
        -------
        MetadataManifest
            Copy of the manifest entries.
        """
        if deep:
            import copy as copy_mod

            new_entries = [
                ManifestEntry(
                    schema_version=e.schema_version,
                    producer_version=e.producer_version,
                    category=e.category,
                    payload=copy_mod.deepcopy(e.payload),
                    step_id=e.step_id,
                    key=e.key,
                    split=e.split,
                )
                for e in self._entries
            ]
        else:
            new_entries = [
                ManifestEntry(
                    schema_version=e.schema_version,
                    producer_version=e.producer_version,
                    category=e.category,
                    payload=dict(e.payload),
                    step_id=e.step_id,
                    key=e.key,
                    split=e.split,
                )
                for e in self._entries
            ]
        return MetadataManifest(entries=new_entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)
