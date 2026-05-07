"""
MZVAV datasource loader with predefined day-level splits.

The raw CSV contains timestamped HVAC measurements and sparse fault labels.
This loader reconstructs the same day-level split logic used historically in
the project:

- known fault dates are annotated with grouped fault ids
- the annotated days are split into train, validation, and test subsets
- all rows belonging to the selected days are returned for each split

When ``group_by_days`` is enabled, the split payload is additionally regrouped
into ragged day-level sequences.
"""

import logging
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import awkward as ak
import pandas as pd
from sklearn.model_selection import train_test_split

from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)
from picid.utils.awkward_utils import ak_unflatten_discontinous_groups

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Stable figshare file IDs for the LBNL MZVAV FDD dataset (CC0).
# Article: https://figshare.com/articles/dataset/LBNLDataSynthesisInventory_pdf/11752740
_FIGSHARE_URLS: dict[str, str] = {
    "MZVAV-1.csv": "https://ndownloader.figshare.com/files/21403008",
    "MZVAV-2-1.csv": "https://ndownloader.figshare.com/files/21403011",
    "MZVAV-2-2.csv": "https://ndownloader.figshare.com/files/21403014",
}


def _download_if_missing(path: Path) -> None:
    """
    Download the MZVAV CSV from figshare if it is not already present.

    Parameters
    ----------
    path : Path
        Expected location of the CSV file (e.g.
        ``<data_dir>/building/MZVAV-2-2.csv``).

    Notes
    -----
    The download is skipped silently when the file already exists.
    If the filename is not in the known figshare registry a warning is emitted
    and the function returns without downloading, allowing downstream code to
    raise a more descriptive ``FileNotFoundError``.
    """
    if path.exists():
        return

    url = _FIGSHARE_URLS.get(path.name)
    if url is None:
        logger.warning(
            "No figshare URL registered for '%s'. Place the file manually at %s.",
            path.name,
            path,
        )
        return

    logger.info("MZVAV file not found at %s. Downloading from figshare ...", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310
            if response.status != 200:
                raise RuntimeError(
                    f"Failed to download MZVAV dataset: HTTP {response.status} from {url}"
                )
            with open(tmp_path, "wb") as f:
                while chunk := response.read(1 << 16):  # 64 KiB chunks
                    f.write(chunk)
        tmp_path.rename(path)
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download MZVAV dataset from {url}: {exc}"
        ) from exc
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("MZVAV file downloaded to %s.", path)


class MZVAVLoader(PredefinedSplitLoaderBase):
    """
    Load the MZVAV dataset and expose train/validation/test day splits.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV file.
    timestamp_name : str
        Name of the timestamp column used as the dataframe index.
    target_name : str
        Name of the sparse fault-label column.
    random_state : int
        Random seed used to keep the day split reproducible.
    group_by_days : bool, default=False
        Whether to regroup each split into ragged day-level sequences.
    **kwargs
        Additional predefined-split loader configuration forwarded to the base
        class.

    Examples
    --------
    A typical configuration provides:

    - ``data_path`` pointing to the CSV file
    - ``timestamp_name`` naming the timestamp column
    - ``target_name`` naming the sparse fault label column
    - ``random_state`` to keep the day split reproducible
    """

    def __init__(
        self,
        data_path,
        timestamp_name,
        target_name,
        random_state,
        group_by_days=False,
        **kwargs,
    ):
        self.data_path = data_path
        self.timestamp_name = timestamp_name
        self.target_name = target_name
        self.group_by_days = group_by_days
        self.random_state = random_state
        super().__init__(
            **kwargs,
        )

    def _load_data(self):
        """
        Read the CSV data and derive day-based predefined splits.

        Returns
        -------
        dict[str, dict[str, pandas.DataFrame | list[awkward.Array]]]
            Split-aware dictionary containing the feature frame, target frame,
            timestamps, initial timestamps, and day identifiers. When
            ``group_by_days`` is enabled, each split field is wrapped into
            day-level Awkward sequences.

        Notes
        -----
        The train/validation/test partitioning is performed on the set of
        labeled days, not on individual rows. Every row belonging to a selected
        day is kept together in the same split.
        """
        df, df_target, df_stamp, days_to_fault = self.read_data()

        train_val_days, test_days = train_test_split(
            days_to_fault,
            test_size=0.2,
            stratify=days_to_fault.values,
            random_state=self.random_state,
        )

        train_days, val_days = train_test_split(
            train_val_days,
            test_size=0.25,
            stratify=train_val_days.values,
            random_state=self.random_state,
        )

        split_dict = {"train": train_days, "val": val_days, "test": test_days}
        data_dict = {
            "features": df,
            "target": df_target,
            "timestamps": df_stamp,
            "initial_dates": df_stamp,
            "days": df.groupby(df.index.date).ngroup(),
        }

        out_dict = {name: {} for name, _ in data_dict.items()}
        for split_key, split_series in split_dict.items():
            for name, data in data_dict.items():
                split_mask = data.index.normalize().isin(split_series.index.normalize())
                out_dict[name][split_key] = data[split_mask].sort_index()

        if self.group_by_days:
            outs_ak = {
                k1: {
                    k2: [
                        ak_unflatten_discontinous_groups(
                            ak.from_numpy(v2.values),
                            data_dict["days"].loc[v2.index].values,
                        )
                    ]
                    for k2, v2 in v1.items()
                }
                for k1, v1 in out_dict.items()
            }
            return outs_ak
        else:
            return out_dict

    def read_data(self):
        """
        Read the raw CSV and derive the day-level fault labels.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
            Feature dataframe, target dataframe, timestamp series, and one
            label per fault day.

        Notes
        -----
        The hard-coded ``fault_number_by_dates`` mapping preserves the existing
        project convention that several calendar dates belong to one grouped
        fault family, for example damper or heating-coil faults.
        """
        _download_if_missing(Path(self.data_path))

        df = pd.read_csv(
            self.data_path,
            sep=",",
            index_col=self.timestamp_name,
            date_format="%m/%d/%Y %H:%M",
        )

        # Forward-fill missing values before labeling.
        df = df.ffill()

        assert df.isna().sum().sum() == 0

        # Mark the known fault dates with their grouped label ids.
        fault_number_by_dates = {
            "2/12/2008": 1,
            "5/7/2008": 1,
            "5/8/2008": 1,
            "9/5/2007": 1,
            "9/6/2007": 1,  # Damper
            "8/28/2007": 2,
            "8/29/2007": 2,
            "8/30/2007": 2,  # Heating Coil
            "5/6/2008": 3,
            "8/31/2007": 3,
            "5/15/2008": 3,
            "9/1/2007": 3,
            "9/2/2007": 3,
        }  # Cooling Coil

        # fault_number_by_dates = {"2/12/2008": 1, "5/7/2008": 1, "5/8/2008": 2, "9/5/2007": 3, "9/6/2007": 4,  # Damper
        #                          "8/28/2007": 5, "8/29/2007": 6, "8/30/2007": 7,                           # Heating Coil
        #                          "5/6/2008": 8, "8/31/2007": 9, "5/15/2008": 9, "9/1/2007": 10, "9/2/2007": 11} # Cooling Coil

        for date, number in fault_number_by_dates.items():
            assert date in df.index
            df.loc[date, self.target_name] = number

        days_to_fault = df[self.target_name].resample("D").first().dropna()

        # Keep the timestamps before removing the target column.
        df_stamp = df.index.to_series().copy()

        # Extract the target and keep it as a dataframe so downstream code sees
        # a ``(-1, 1)`` shape instead of a flat series.
        df_target = df[self.target_name].to_frame().copy()
        df.drop(columns=[self.target_name], inplace=True)

        return df, df_target, df_stamp, days_to_fault
