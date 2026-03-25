"""N-CMAPSS datasource loader used for concept-aware experiments.

This loader reads one ``N-CMAPSS_DS*.h5`` file and returns the aligned sensor
signals, operating conditions, health-state metadata, concept targets, and RUL
targets expected by the concept-learning experiments in the repository.

The implementation supports the same configuration knobs that were already used
throughout the project, such as:

- selecting one dataset id through ``load_arguments.n_DS``
- restricting the payload either to a set of units or to one flight class
- optionally keeping only cruise segments
- optionally flattening the healthy prefix of the RUL target
- optionally grouping the final arrays by unit into ragged Awkward Arrays
"""

import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import awkward as ak

from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.utils.awkward_utils import ak_unflatten_discontinous_groups

logger = logging.getLogger(__name__)


def subsampling(df, subsampling_rate):
    """Reduce computational cost by subsampling the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to subsample.
    subsampling_rate : int
        Step size used to keep every ``subsampling_rate`` row.

    Returns
    -------
    pandas.DataFrame
        Subsampled dataframe view.

    Examples
    --------
    Keeping every tenth row is equivalent to::

        subsampling(df, 10)
    """
    return df[::subsampling_rate]


def binarize_concept(x):
    """Convert a continuous concept signal into a binary degradation flag.

    The original implementation treated sufficiently negative concept values as
    active degradation states. This helper keeps that thresholding logic in one
    place.
    """
    return x < -0.0015


def scale_concept(x):
    """Scale a continuous concept signal into the legacy ``[0, 1]`` range.

    Notes
    -----
    The scaling follows the legacy convention used by the existing BCE-style
    concept-learning setups.
    """
    return np.clip(x / -0.035, 0, 1)


def flatten_RUL(df):
    """Flatten the healthy prefix of the RUL signal before the first HS drop.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-unit dataframe containing at least ``hs``, ``cycle``, and ``RUL``.

    Returns
    -------
    pandas.DataFrame
        Dataframe where the prefix before the first ``hs`` transition from
        ``1`` to ``0`` shares the same RUL value.

    Examples
    --------
    If the first transition happens at cycle ``k``, every earlier row gets the
    RUL value observed at cycle ``k``. This preserves the legacy "flat RUL"
    behavior used in existing experiments.
    """
    hs_change = df[df["hs"].diff() == -1]
    if len(hs_change) > 0:
        cutoff_cycle = hs_change.iloc[0]["cycle"]
        cutoff_rul = hs_change.iloc[0]["RUL"]
        df.loc[df["cycle"] < cutoff_cycle, "RUL"] = cutoff_rul
    return df


class N_CMAPSSDataSource(SingleSourceLoader):
    """Datasource for the N-CMAPSS dataset with optional concept targets.

    The returned payload combines:

    - ``features``: sensor measurements
    - ``descriptors``: operating conditions
    - ``concepts``: degradation concepts derived from the PHM challenge signals
    - ``rul``: remaining useful life target
    - ``health_states`` and ``unit``: auxiliary grouping information

    Depending on ``group_by_unit``, the payload is returned either as flat NumPy
    arrays aligned row by row or as ragged Awkward Arrays grouped by unit.
    """

    def __init__(
        self,
        path: str,
        load_arguments: dict[str, Any],
        RUL: str = "linear",
        cruise: bool = False,
        subsampling_rate: int = 1,
        concepts: list[str] = ["LPT", "HPT"],
        combined_concepts: bool = False,
        binary_concepts: bool = True,
        include_healthy: bool = True,
        scaling: str = "legacy",
        group_by_unit: bool = False,
        **kwargs: Any,
    ):
        """Initialise the datasource-specific loading configuration.

        Parameters
        ----------
        path : str
            Directory containing the ``N-CMAPSS_DS*.h5`` files.
        load_arguments : dict[str, Any]
            Hydra-style configuration describing the requested mode, dataset,
            and optional unit or flight-class filters.
        RUL : str, default="linear"
            Remaining useful life target style. Supported values are
            ``"linear"`` and ``"flat"``.
        cruise : bool, default=False
            If ``True``, keep only the cruise phase for each flight cycle.
        subsampling_rate : int, default=1
            Optional subsampling factor applied after filtering.
        concepts : list[str], default=["LPT", "HPT"]
            Concept targets requested by the experiment configuration.
        combined_concepts : bool, default=False
            Whether pairwise combined concept labels should be added.
        binary_concepts : bool, default=True
            Whether concepts are emitted as binary flags instead of scaled
            continuous values.
        include_healthy : bool, default=True
            Whether healthy rows should remain in the returned payload.
        scaling : str, default="legacy"
            Legacy scaling mode kept for compatibility with existing configs.
        group_by_unit : bool, default=False
            Whether the final payload should be grouped into ragged per-unit
            sequences.
        **kwargs : Any
            Additional datasource-base keyword arguments.

        Notes
        -----
        Exactly one of ``load_arguments.units`` or ``load_arguments.fc`` may be
        provided. This preserves the existing convention that a configuration
        either selects explicit units or one flight class, but never both.

        Examples
        --------
        A typical standalone configuration sets::

            load_arguments.mode = "train"
            load_arguments.n_DS = "02"
            load_arguments.units = [2, 5, 10]

        while leaving ``fc`` unset.
        """
        assert RUL in (
            "linear",
            "flat",
        ), "RUL type must be 'linear' or 'flat"

        logger.info(f"Using {RUL} RUL.")
        self.mode = load_arguments.mode
        self.n_DS = load_arguments.n_DS

        self.selected_units = load_arguments.get("units", None)
        self.selected_fc = load_arguments.get("fc", None)

        assert self.selected_fc in (1, 2, 3, None), "fc must be 1, 2, 3 or None"
        if not ((self.selected_units is None) ^ (self.selected_fc is None)):
            raise ValueError("Specify exactly one of fc or units")

        self.RUL = RUL
        self.path = path
        self.include_healthy = include_healthy
        self.subsampling_rate = subsampling_rate
        self.cruise = cruise
        self.requested_concepts = list(load_arguments.concepts)
        self.binary_concepts = binary_concepts
        self.combined_concepts = combined_concepts
        self.scaling = scaling
        self.group_by_unit = group_by_unit

        super().__init__(**kwargs)

    def _load_data(self):
        """Load the configured N-CMAPSS split and derive concept features.

        Returns
        -------
        dict[str, numpy.ndarray | awkward.Array]
            Dictionary containing the aligned arrays used by downstream
            preprocessing. When ``group_by_unit`` is enabled, each field is
            returned as a ragged Awkward Array grouped by unit.

        Notes
        -----
        The loader keeps the established processing order:

        1. read the requested ``dev`` or ``test`` tensors from HDF5
        2. apply optional RUL flattening
        3. filter healthy rows, units, or flight classes if requested
        4. optionally keep only cruise segments
        5. derive concept columns
        6. optionally regroup by unit
        """

        filename = os.path.join(
            Path(self.path).expanduser(), f"N-CMAPSS_DS{self.n_DS}.h5"
        )
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        with h5py.File(filename, "r") as hdf:
            if "train" in self.mode or "val" in self.mode:
                # Development split tensors.
                W = np.array(hdf.get("W_dev"))  # W
                X_s = np.array(hdf.get("X_s_dev"))  # X_s
                _X_v = np.array(hdf.get("X_v_dev"))  # X_v
                T = np.array(hdf.get("T_dev"))  # T
                Y = np.array(hdf.get("Y_dev"))  # RUL
                A = np.array(hdf.get("A_dev"))  # Auxiliary

            elif "test" in self.mode:
                # Test split tensors.
                W = np.array(hdf.get("W_test"))  # W
                X_s = np.array(hdf.get("X_s_test"))  # X_s
                _X_v = np.array(hdf.get("X_v_test"))  # X_v
                T = np.array(hdf.get("T_test"))  # T
                Y = np.array(hdf.get("Y_test"))  # RUL
                A = np.array(hdf.get("A_test"))  # Auxiliary

            # Variable names for each tensor block.
            W_var = np.array(hdf.get("W_var"))
            X_s_var = np.array(hdf.get("X_s_var"))
            X_v_var = np.array(hdf.get("X_v_var"))
            T_var = np.array(hdf.get("T_var"))
            A_var = np.array(hdf.get("A_var"))

            # Convert HDF string arrays into normal Python string lists.
            W_var = list(np.array(W_var, dtype="U20"))
            X_s_var = list(np.array(X_s_var, dtype="U20"))
            X_v_var = list(np.array(X_v_var, dtype="U20"))
            T_var = list(np.array(T_var, dtype="U20"))
            A_var = list(np.array(A_var, dtype="U20"))

        # TODO: Concatenate sensor data if virtual sensors should be considered.
        # if sensors == 'all':
        #     X = np.concatenate((X_s, X_v), axis=-1)
        # else:
        #     X = X_s

        # Auxiliary data: ``unit``, ``cycle``, ``Fc``, and ``hs``.
        self.df_A = pd.DataFrame(data=A, columns=A_var).astype(int)
        self.units = list(np.unique(self.df_A["unit"]))

        # Operating conditions: ``alt``, ``Mach``, ``TRA``, and ``T2``.
        self.df_W = pd.DataFrame(data=W, columns=W_var)

        # Degradation variables used to derive concepts.
        self.df_T = pd.DataFrame(data=T, columns=T_var)

        # Sensor measurements.
        self.df_X = pd.DataFrame(data=X_s, columns=X_s_var)

        # Remaining useful life target.
        self.df_Y = pd.DataFrame(data=Y, columns=["RUL"])

        del A, W, T, X_s, Y

        if self.RUL == "flat":
            # Keep the healthy prefix at a flat value before the first HS drop.
            df_all = pd.concat((self.df_A, self.df_Y), axis=1)
            self.df_Y["RUL"] = df_all.groupby("unit", group_keys=False).apply(
                flatten_RUL
            )["RUL"]
            del df_all

        if not self.include_healthy:
            self.df_W = self.df_W.loc[self.df_A["hs"] == 0]
            self.df_T = self.df_T.loc[self.df_A["hs"] == 0]
            self.df_X = self.df_X.loc[self.df_A["hs"] == 0]
            self.df_Y = self.df_Y.loc[self.df_A["hs"] == 0]
            self.df_A = self.df_A.loc[self.df_A["hs"] == 0]

        if self.selected_units is not None:
            # Keep only the requested unit subset.
            self.df_W = self.df_W.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_T = self.df_T.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_X = self.df_X.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_Y = self.df_Y.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_A = self.df_A.loc[self.df_A["unit"].isin(self.selected_units)]

        if self.selected_fc is not None:
            # Keep only the requested flight-class subset.
            self.df_W = self.df_W.loc[self.df_A["Fc"] == self.selected_fc]
            self.df_T = self.df_T.loc[self.df_A["Fc"] == self.selected_fc]
            self.df_X = self.df_X.loc[self.df_A["Fc"] == self.selected_fc]
            self.df_Y = self.df_Y.loc[self.df_A["Fc"] == self.selected_fc]
            self.df_A = self.df_A.loc[self.df_A["Fc"] == self.selected_fc]

        if self.subsampling_rate > 1:
            # Subsample every aligned dataframe with the same stride.
            self.df_A = subsampling(self.df_A, self.subsampling_rate)
            self.df_W = subsampling(self.df_W, self.subsampling_rate)
            self.df_X = subsampling(self.df_X, self.subsampling_rate)
            self.df_T = subsampling(self.df_T, self.subsampling_rate)
            self.df_Y = subsampling(self.df_Y, self.subsampling_rate)

        if self.cruise:
            # For each unit and cycle, keep only the cruising regime.
            df_all = pd.concat((self.df_A, self.df_W), axis=1)
            df_cruise = df_all.groupby(["unit", "cycle"], group_keys=False).apply(
                lambda flight: flight[flight["alt"] >= flight["alt"].max() - 1000]
            )
            self.df_X = self.df_X.loc[df_cruise.index]
            self.df_T = self.df_T.loc[df_cruise.index]
            self.df_Y = self.df_Y.loc[df_cruise.index]
            self.df_A = self.df_A.loc[df_cruise.index]
            self.df_W = self.df_W.loc[df_cruise.index]
            del df_all, df_cruise

        requested_concepts: str | list[str]
        if len(self.requested_concepts) == 1 and self.requested_concepts[0] == "all":
            requested_concepts = [
                f"{c}-{m}"
                for c in ("Fan", "LPC", "HPC", "LPT", "HPT")
                for m in ("E", "F")
            ]
        else:
            requested_concepts = list(self.requested_concepts)

        concepts = pd.DataFrame(
            {
                "Fan": self.df_T[["fan_eff_mod", "fan_flow_mod"]].min(axis=1),
                "LPC": self.df_T[["LPC_eff_mod", "LPC_flow_mod"]].min(axis=1),
                "HPC": self.df_T[["HPC_eff_mod", "HPC_flow_mod"]].min(axis=1),
                "LPT": self.df_T[["LPT_eff_mod", "LPT_flow_mod"]].min(axis=1),
                "HPT": self.df_T[["HPT_eff_mod", "HPT_flow_mod"]].min(axis=1),
                "Fan-E": self.df_T["fan_eff_mod"],
                "Fan-F": self.df_T["fan_flow_mod"],
                "LPC-E": self.df_T["LPC_eff_mod"],
                "LPC-F": self.df_T["LPC_flow_mod"],
                "HPC-E": self.df_T["HPC_eff_mod"],
                "HPC-F": self.df_T["HPC_flow_mod"],
                "LPT-E": self.df_T["LPT_eff_mod"],
                "LPT-F": self.df_T["LPT_flow_mod"],
                "HPT-E": self.df_T["HPT_eff_mod"],
                "HPT-F": self.df_T["HPT_flow_mod"],
            }
        )[[c for c in requested_concepts if c not in ["healthy", "Fc"]]]

        # Remove non-minimum degradation.
        # concepts = concepts.apply(lambda row: pd.Series([val if val == min(row) else 0 for val in row]), axis=1)

        if self.binary_concepts:
            concepts = concepts.apply(binarize_concept)
        else:  # Continuous concepts in ``[0, 1]`` for BCE-style losses.
            concepts = concepts.apply(scale_concept)
        if (
            self.combined_concepts
        ):  # TODO: Treat the continuous-concept case explicitly.
            for combination in combinations(concepts.columns.tolist(), 2):
                combination_col = "+".join(combination)
                concepts[combination_col] = concepts[list(combination)].all(axis=1)
                for c in combination:
                    concepts[c] = np.logical_xor(concepts[c], concepts[combination_col])
        if "healthy" in requested_concepts:
            concepts["healthy"] = (concepts == 0).all(axis=1)
        if "Fc" in requested_concepts:
            concepts = pd.concat(
                (
                    concepts,
                    pd.get_dummies(self.df_A["Fc"]).reindex(
                        columns=[1, 2, 3], fill_value=0
                    ),
                ),
                axis=1,
            )
        # drop constant concepts
        # concepts = concepts.loc[:, (concepts != concepts.iloc[0]).any()]
        logger.info(f"Used concepts: {concepts.columns}")
        concepts = concepts.astype(int)

        # ak.from_arrow(pa.Table.from_pandas(self.df_X))

        outs = {
            "features": self.df_X.values,  # Sensor measurements.
            "timestamps": self.df_X.index.values.reshape(-1, 1),  # Cycle numbers.
            "health_states": self.df_A["hs"].values.reshape(-1, 1),
            "unit": self.df_A["unit"].values,
            "n_DS": np.ones(self.df_A["unit"].shape) * int(self.n_DS),
            "descriptors": self.df_W.values,  # Operating conditions.
            "concepts": concepts.values,
            "rul": self.df_Y.values,  # Remaining useful life.
            # degradation=self.df_T.values, # Degradation variables behind the concepts.
            # auxiliary_data=self.df_A.values, # Auxiliary data: ``unit``, ``cycle``, ``Fc``, ``hs``.
        }

        if self.group_by_unit:
            outs_ak = {k: ak.from_numpy(v) for k, v in outs.items()}
            outs_ak = {
                k: ak_unflatten_discontinous_groups(v, outs["unit"])
                for k, v in outs_ak.items()
            }
            return outs_ak
        else:
            return outs
