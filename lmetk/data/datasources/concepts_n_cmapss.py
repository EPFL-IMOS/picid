
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from lmetk.data.datasources.base import DataSourceLoader

from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations

from lmetk.data.preprocessing.scalers import MultiSourceTransformMixin

from lmetk.transforms.base_transform import BaseTransform
from typing import Any, List, Union, Tuple, Type

import logging
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


class N_CMAPSSScaler_Descriptors(BaseTransform): #BaseEstimator, TransformerMixin, MultiSourceTransformMixin):

    def __init__(self, scaling="standard"):

        super().__init__()
        self.scaling = scaling


    # def fit(self, W, y=None):
    #     pass  # No fitting needed for this scaler
    def fit_single_source(self, data_segment: Any):
        pass

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        pass

    def transform_data(self, list_of_data_segments: List[np.array], mode: str):
        scaling = self.scaling
        if scaling == "standard":
            W_mean_1457 = np.array(
                [1.6362398e04, 5.4544175e-01, 6.1369926e01, 4.8835443e02]
            )
            W_std_1457 = np.array(
                [8.1254497e03, 1.2108228e-01, 1.8272049e01, 1.9934254e01]
            )
            list_of_data_segments = [ (data_segment - W_mean_1457) / W_std_1457 for data_segment in list_of_data_segments]
            #W = (W - W_mean_1457) / W_std_1457


        elif scaling == "min-max":
            # min-max scaling in [-1, 1] range
            W_min_1457 = np.array(
                [3.0020000e03, 2.0002499e-01, 2.3730299e01, 4.2319705e02]
            )
            W_max_1457 = np.array(
                [3.5011000e04, 7.3987198e-01, 8.7362656e01, 5.2488281e02]
            )
            list_of_data_segments = [ 2 * (data_segment - W_min_1457) / (W_max_1457 - W_min_1457) - 1 for data_segment in list_of_data_segments]
            #W = 2 * (W - W_min_1457) / (W_max_1457 - W_min_1457) - 1
        return list_of_data_segments


class N_CMAPSSScaler_Features(BaseTransform): #BaseEstimator, TransformerMixin, MultiSourceTransformMixin):

    def __init__(self, scaling="standard"):
        self.scaling = scaling


    def fit_single_source(self, data_segment: Any):
        pass

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        pass

    def transform_data(self, list_of_data_segments: List[np.array], mode: str):
        scaling = self.scaling
        if scaling == "standard":
            # standard scaling
            X_mean_1457 = np.array(
                [
                    5.6809174e02,
                    1.3297987e03,
                    1.6363848e03,
                    1.1242614e03,
                    1.2669851e01,
                    9.8758097e00,
                    1.2862692e01,
                    1.5640701e01,
                    2.3289339e02,
                    2.3705125e02,
                    9.8489847e00,
                    1.9629447e03,
                    8.2361680e03,
                    2.4945383e00,
                ]
            )

            X_std_1457 = np.array(
                [
                    2.0833166e01,
                    6.7058266e01,
                    1.2113522e02,
                    6.1629440e01,
                    2.8704438e00,
                    2.4181337e00,
                    2.9141707e00,
                    3.4177959e00,
                    5.7780758e01,
                    5.8610283e01,
                    2.7480240e00,
                    1.8454973e02,
                    2.2242123e02,
                    7.6335633e-01,
                ]
            )
            list_of_data_segments = [ (data_segment - X_mean_1457) / X_std_1457 for data_segment in list_of_data_segments]
            


        elif scaling == "min-max":
            # min-max scaling in [-1, 1] range
            X_min_1457 = np.array(
                [
                    4.9921100e02,
                    1.0889111e03,
                    1.2294357e03,
                    9.1258221e02,
                    6.0976486e00,
                    4.4431176e00,
                    6.1691909e00,
                    7.8722315e00,
                    9.1395287e01,
                    9.3320122e01,
                    4.5537462e00,
                    1.4843353e03,
                    7.4330903e03,
                    7.4892521e-01,
                ]
            )

            X_max_1457 = np.array(
                [
                    6.3150879e02,
                    1.5320127e03,
                    1.9813180e03,
                    1.3461113e03,
                    2.0096666e01,
                    1.4716009e01,
                    2.0402708e01,
                    2.5905502e01,
                    4.4721826e02,
                    4.5370425e02,
                    1.6701591e01,
                    2.2791799e03,
                    8.8905791e03,
                    5.6173835e00,
                ]
            )
            list_of_data_segments = [ 2 * (data_segment - X_min_1457) / (X_max_1457 - X_min_1457) - 1 for data_segment in list_of_data_segments]
            #X = 2 * (X - X_min_1457) / (X_max_1457 - X_min_1457) - 1
        return list_of_data_segments

class ConstantScaler(BaseTransform):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def fit_single_source(self, data_segment: Any):
        pass

    def fit_multi_source(self, list_of_data_segments: List[Any]):
        pass

    def transform_data(self, list_of_data_segments: List[np.array], mode: str):
        
        return [ data_segment * self.factor for data_segment in list_of_data_segments] 

class N_CMAPSSDataSource(DataSourceLoader):

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

    def _load_data(self):
        # Load the data from the specified path
        assert self.RUL in ("linear", "flat"), "RUL type must be 'linear' or 'flat"
        logger.info(f"Using {self.RUL} RUL.")
        self.n_DS = self.n_DS
        self.mode = self.mode
        self.window_size = self.window_size
        self.stride = self.stride

        filename = os.path.join(Path(self.path).expanduser(), f"N-CMAPSS_DS{self.n_DS}.h5")
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

        if self.units is not None:
            # select only unit data
            self.df_W = self.df_W.loc[self.df_A["unit"].isin(self.units)]
            self.df_T = self.df_T.loc[self.df_A["unit"].isin(self.units)]
            self.df_X = self.df_X.loc[self.df_A["unit"].isin(self.units)]
            self.df_Y = self.df_Y.loc[self.df_A["unit"].isin(self.units)]
            self.df_A = self.df_A.loc[self.df_A["unit"].isin(self.units)]

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

        
         
        return { 
            "features": self.df_X.values, # sensor data measurements
            "timestamps": self.df_X.index.values.reshape(-1,1),  # cycle numbers
            "health_states": self.df_A["hs"].values,
            'unit': self.df_A["unit"].values,
            "descriptors": self.df_W.values, # operating conditions
            "concepts": self.concepts.values,
            "rul":self.df_Y.values, # RUL
            # degradation=self.df_T.values, # degradation resp. concepts
            # auxiliary_data=self.df_A.values, # aux data: 'unit', 'cycle', 'Fc', 'hs'
        }