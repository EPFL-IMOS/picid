import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import awkward as ak
import pyarrow as pa

from lmetk.data.datasources import SingleSourceLoader
from lmetk.utils.awkward_utils import ak_unflatten_discontinous_groups

logger = logging.getLogger(__name__)


def subsampling(df, subsampling_rate):
    """
    reduce computational cost by subsampling the data
    :param df: pd.pd.DataFrame, subsampling data
    :param subsampling_rate: int, subsampling rate, reduce size to 1/subsampling_rate
    """
    return df[::subsampling_rate]


def binarize_concept(x):
    return x < -0.0015


def scale_concept(x):
    return np.clip(x / -0.035, 0, 1)


def flatten_RUL(df):
    hs_change = df[df["hs"].diff() == -1]
    if len(hs_change) > 0:
        df["RUL"].loc[df["cycle"] < hs_change.iloc[0]["cycle"]] = hs_change.iloc[0][
            "RUL"
        ]
    return df


class N_CMAPSSDataSource(SingleSourceLoader):

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
        window_size: int = 50,
        stride: int = 1,
        scaling: str = "legacy",
        group_by_unit: bool = False,
        **kwargs: Any,
    ):

        assert RUL in (
            "linear",
            "flat",
        ), "RUL type must be 'linear' or 'flat"

        logger.info(f"Using {RUL} RUL.")
        self.mode = load_arguments.mode
        self.n_DS = load_arguments.n_DS
        self.selected_units = load_arguments.units

        self.RUL = RUL
        self.path = path
        self.window_size = window_size
        self.stride = stride
        self.include_healthy = include_healthy
        self.subsampling_rate = subsampling_rate
        self.cruise = cruise
        self.concepts = concepts
        self.binary_concepts = binary_concepts
        self.combined_concepts = combined_concepts
        self.scaling = scaling
        self.group_by_unit = group_by_unit

        super().__init__(**kwargs)

    def _load_data(self):
        # Load the data from the specified path

        filename = os.path.join(
            Path(self.path).expanduser(), f"N-CMAPSS_DS{self.n_DS}.h5"
        )
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        with h5py.File(filename, "r") as hdf:

            if "train" in self.mode or "val" in self.mode:
                # Development set
                W = np.array(hdf.get("W_dev"))  # W
                X_s = np.array(hdf.get("X_s_dev"))  # X_s
                X_v = np.array(hdf.get("X_v_dev"))  # X_v
                T = np.array(hdf.get("T_dev"))  # T
                Y = np.array(hdf.get("Y_dev"))  # RUL
                A = np.array(hdf.get("A_dev"))  # Auxiliary

            elif "test" in self.mode:
                # Test set
                W = np.array(hdf.get("W_test"))  # W
                X_s = np.array(hdf.get("X_s_test"))  # X_s
                X_v = np.array(hdf.get("X_v_test"))  # X_v
                T = np.array(hdf.get("T_test"))  # T
                Y = np.array(hdf.get("Y_test"))  # RUL
                A = np.array(hdf.get("A_test"))  # Auxiliary

            # Varnams
            W_var = np.array(hdf.get("W_var"))
            X_s_var = np.array(hdf.get("X_s_var"))
            X_v_var = np.array(hdf.get("X_v_var"))
            T_var = np.array(hdf.get("T_var"))
            A_var = np.array(hdf.get("A_var"))

            # from np.array to list dtype U4/U5
            W_var = list(np.array(W_var, dtype="U20"))
            X_s_var = list(np.array(X_s_var, dtype="U20"))
            X_v_var = list(np.array(X_v_var, dtype="U20"))
            T_var = list(np.array(T_var, dtype="U20"))
            A_var = list(np.array(A_var, dtype="U20"))

        # TODO: concat sensor data if virtual senssors should be considered
        # if sensors == 'all':
        #     X = np.concatenate((X_s, X_v), axis=-1)
        # else:
        #     X = X_s

        # aux data: 'unit', 'cycle', 'Fc', 'hs'
        self.df_A = pd.DataFrame(data=A, columns=A_var).astype(int)
        self.units = list(np.unique(self.df_A["unit"]))

        # operating conditions: 'alt', 'Mach', 'TRA', 'T2'
        self.df_W = pd.DataFrame(data=W, columns=W_var)

        # degradation resp. concepts
        self.df_T = pd.DataFrame(data=T, columns=T_var)

        # sensor data
        self.df_X = pd.DataFrame(data=X_s, columns=X_s_var)

        # RUL
        self.df_Y = pd.DataFrame(data=Y, columns=["RUL"])

        del A, W, T, X_s, Y

        if self.RUL == "flat":
            # Set RUL to flat constant value when hs == 1
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
            # select only unit data
            self.df_W = self.df_W.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_T = self.df_T.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_X = self.df_X.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_Y = self.df_Y.loc[self.df_A["unit"].isin(self.selected_units)]
            self.df_A = self.df_A.loc[self.df_A["unit"].isin(self.selected_units)]

        if self.subsampling_rate > 1:
            # subsample
            self.df_A = subsampling(self.df_A, self.subsampling_rate)
            self.df_W = subsampling(self.df_W, self.subsampling_rate)
            self.df_X = subsampling(self.df_X, self.subsampling_rate)
            self.df_T = subsampling(self.df_T, self.subsampling_rate)
            self.df_Y = subsampling(self.df_Y, self.subsampling_rate)

        if self.cruise:
            # for each unit and cycle, get only 'cruising' state
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

        if self.concepts == "all":
            self.concepts = [
                f"{c}-{m}"
                for c in ("Fan", "LPC", "HPC", "LPT", "HPT")
                for m in ("E", "F")
            ]

        self.concepts = pd.DataFrame(
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
        )[[c for c in self.concepts if c not in ["healthy", "Fc"]]]

        # remove non-minimum degradation
        # self.concepts = self.concepts.apply(lambda row: pd.Series([val if val == min(row) else 0 for val in row]), axis=1)

        if self.binary_concepts:
            self.concepts = self.concepts.apply(binarize_concept)
        else:  # continuous concepts (in [0,1] for BCE loss)
            self.concepts = self.concepts.apply(scale_concept)
        if (
            self.combined_concepts
        ):  # TODO: treat the case with continuous concepts (use mean instead of all?)
            for combination in combinations(self.concepts.columns.tolist(), 2):
                combination_col = "+".join(combination)
                self.concepts[combination_col] = self.concepts[list(combination)].all(
                    axis=1
                )
                for c in combination:
                    self.concepts[c] = np.logical_xor(
                        self.concepts[c], self.concepts[combination_col]
                    )
        if "healthy" in self.concepts:
            self.concepts["healthy"] = (self.concepts == 0).all(axis=1)
        if "Fc" in self.concepts:
            self.concepts = pd.concat(
                (
                    self.concepts,
                    pd.get_dummies(self.df_A["Fc"]).reindex(
                        columns=[1, 2, 3], fill_value=0
                    ),
                ),
                axis=1,
            )
        # drop constant concepts
        # self.concepts = self.concepts.loc[:, (self.concepts != self.concepts.iloc[0]).any()]
        logger.info(f"Used concepts: {self.concepts.columns}")
        self.concepts = self.concepts.astype(int)

        # ak.from_arrow(pa.Table.from_pandas(self.df_X))

        outs = {
            "features": self.df_X.values,  # sensor data measurements
            "timestamps": self.df_X.index.values.reshape(-1, 1),  # cycle numbers
            "health_states": self.df_A["hs"].values,
            "unit": self.df_A["unit"].values,
            "descriptors": self.df_W.values,  # operating conditions
            "concepts": self.concepts.values,
            "rul": self.df_Y.values,  # RUL
            # degradation=self.df_T.values, # degradation resp. concepts
            # auxiliary_data=self.df_A.values, # aux data: 'unit', 'cycle', 'Fc', 'hs'
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
