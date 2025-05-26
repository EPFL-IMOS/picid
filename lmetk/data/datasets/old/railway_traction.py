import os
import sys

from sklearn.preprocessing import MinMaxScaler

from lmetk.case_studies.common import (
    ContextBatchDataset,
    check_boolean_vector,
    time_features,
)
from lmetk.case_studies.converters import dl_to_sklearn__with_context_for_tabpfn

os.path.dirname(sys.executable)

from pathlib import Path
from omegaconf import OmegaConf
import importlib

# from main_nolightning import initialize, configure
import pandas as pd
import numpy as np
from einops import rearrange
import torch
import pytorch_lightning as pl
from typing import Optional, List, Tuple
from abc import abstractmethod
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler

from typing import Optional, List, Tuple
from abc import abstractmethod
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler

from sklearn.preprocessing import MinMaxScaler


class SpatialDataModule(pl.LightningDataModule):
    def __init__(
        self,
        test_ratio: float = 0.2,
        val_ratio: float = 0.07,
        train_ratio: float = 1.0,
        train_scaled: Optional[None] = None,
        val_scaled: Optional[None] = None,
        test_scaled: Optional[None] = None,
        batch_size=64,
        samples_per_day=24,
        trend_decomp_rolling_window=384,
        ts_in=24 * 7,
        ts_out=24,
        step_train=1,
        step_val=24,
        step_test=24,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        data_split_strategy="scale_training_set",
        include_bl_forecast=False,
        datetime_features=True,
        forecast_feature_name="Bahnlast",
        forecast_all_features=False,
        trend_decomp="disabled",
        subsample_strategy="mean_1h",
        spatial_aggregation: Optional[str] = None,
    ):
        super().__init__()
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = train_ratio

        self.samples_per_day = samples_per_day
        self.trend_decomp_rolling_window = trend_decomp_rolling_window
        self.subsample_strategy = subsample_strategy

        self.train_scaled = train_scaled
        self.val_scaled = val_scaled
        self.test_scaled = test_scaled
        self.batch_size = batch_size
        self.ts_in = ts_in
        self.ts_out = ts_out
        self.step_train = step_train
        self.step_val = step_val
        self.step_test = step_test
        self.forecast_feature_name = forecast_feature_name
        self.forecast_all_features = forecast_all_features
        self.data_split_strategy = data_split_strategy

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.datetime_features = datetime_features

        self.num_workers = num_workers
        self.include_bl_forecast = include_bl_forecast
        self.trend_decomp = trend_decomp
        self.spatial_aggregation = spatial_aggregation

    def setup(self, stage: Optional[str] = None):
        if self.train_scaled is None:
            data = pd.read_csv("datasets/railway/railway.csv", sep=",")
            data.datetime = pd.to_datetime(data.datetime)
            data.set_index("datetime", inplace=True)
            self.data = {"full": data}

            samples_per_day = self.samples_per_day

            if self.trend_decomp == "disabled":
                bl = data["Bahnlast"]
                data.rename(columns={"Bahnlast": "Bahnlast_oldpos"}, inplace=True)
                data.insert(loc=0, column="Bahnlast", value=bl)
                data.drop(columns=["Bahnlast_oldpos"], inplace=True)
                self.data = {"full": data, "gt": bl.copy()}

            elif self.trend_decomp == "mean":
                S = self.trend_decomp_rolling_window
                self.prev_mean = data["Bahnlast"].rolling(S).mean().bfill()
                self.prev_mean_shifted = (
                    data["Bahnlast"].rolling(S).mean().shift(samples_per_day).bfill()
                )

                decomp = data["Bahnlast"] - self.prev_mean_shifted
                data.insert(loc=0, column="Bahnlast_decomp", value=decomp)

                self.data = {"full": data, "gt": data["Bahnlast"].copy()}
                data.drop(columns=["Bahnlast"], inplace=True)
            else:
                raise ValueError("Unknown trend decomposition")

            if not self.forecast_all_features:
                self.bahnlast_mask = data.columns == self.forecast_feature_name
                self.bahnlast_mask_orig = data.columns == self.forecast_feature_name
            else:
                self.bahnlast_mask = np.ones(data.shape[1], dtype=bool)
                self.bahnlast_mask_orig = np.ones(data.shape[1], dtype=bool)

            test_ratio = self.test_ratio
            val_ratio = self.val_ratio
            train_ratio = self.train_ratio

            cut_to_days = lambda x: (x // samples_per_day) * samples_per_day
            N_samples = cut_to_days(len(data))
            if self.data_split_strategy == "default":
                N_test = cut_to_days(round(test_ratio * N_samples))
                N_val = cut_to_days(round(val_ratio * N_samples))
                N_train = cut_to_days(N_samples - N_test - N_val)

                r_train = range(0, N_train)
                r_val = range(N_train, N_train + N_val)
                r_test = range(N_train + N_val, N_train + N_val + N_test)
                r_test_small = range(
                    data.index.get_loc("2023-01-24 00:00:00"),
                    data.index.get_loc("2023-07-31 00:00:00"),
                )

            elif self.data_split_strategy == "scale_training_set":
                ################################################
                #    Train  |  Val  |  Test_small  | Test
                #       xxxx|xxxxxxx|xxxxxxxxxxxxxxxxxxxxx
                #   xxxxxxxx|xxxxxxx|xxxxxxxxxxxxxxxxxxxxx train
                #
                ################################################

                # we keep a relatively long test set
                r_test_small = range(
                    data.index.get_loc("2023-01-24 00:00:00"),
                    data.index.get_loc("2023-07-31 00:00:00"),
                )
                r_test = range(data.index.get_loc("2023-01-24 00:00:00"), N_samples)

                # we cut a small validation set
                N_val = cut_to_days(round(val_ratio * (N_samples - len(r_test))))
                Len_train = N_samples - N_val - len(r_test)
                N_train = cut_to_days(round(train_ratio * (Len_train)))

                r_train = range(Len_train - N_train, Len_train)
                r_val = range(Len_train, Len_train + N_val)

            elif self.data_split_strategy == "covid_training_set":
                # we keep a relatively long test set
                r_test_small = range(
                    data.index.get_loc("2023-01-24 00:00:00"),
                    data.index.get_loc("2023-07-31 00:00:00"),
                )
                r_test = range(data.index.get_loc("2023-01-24 00:00:00"), N_samples)

                # we cut a small validation set
                N_val = cut_to_days(round(val_ratio * (N_samples - len(r_test))))
                Len_train = N_samples - N_val - len(r_test)

                r_train = range(
                    data.index.get_loc("2019-12-01 00:00:00"),
                    data.index.get_loc("2021-07-31 00:00:00"),
                )
                r_val = range(Len_train, Len_train + N_val)

            self.r_train, self.r_val, self.r_test, self.r_test_small = (
                r_train,
                r_val,
                r_test,
                r_test_small,
            )
            self.train = data.iloc[r_train, :]
            self.val = data.iloc[r_val, :]
            self.test = data.iloc[r_test, :]
            self.test_small = data.iloc[r_test_small, :]

            scaler = MinMaxScaler()
            scaler.fit(self.train)
            self.train_scaled = scaler.transform(self.train)
            self.val_scaled = scaler.transform(self.val)
            self.test_scaled = scaler.transform(self.test)
            self.test_small_scaled = scaler.transform(self.test_small)

            if self.datetime_features:
                dtf = time_features(
                    pd.Series(data.index),
                    use_features=["month", "day", "weekday", "hour"],
                )
                self.dtf_train = dtf.iloc[r_train, :]
                self.dtf_val = dtf.iloc[r_val, :]
                self.dtf_test = dtf.iloc[r_test, :]
                self.dtf_test_small = dtf.iloc[r_test_small, :]

                self.scaler_mask = np.concatenate(
                    (
                        np.ones(self.train_scaled.shape[1], dtype=bool),
                        np.zeros(4, dtype=bool),
                    )
                )

                self.bahnlast_mask = np.concatenate(
                    (self.bahnlast_mask, np.zeros(4, dtype=bool))
                )

                self.train_scaled = np.concatenate(
                    (
                        self.train_scaled,
                        self.dtf_train[["hour", "day", "weekday", "month"]].values,
                    ),
                    axis=1,
                )
                self.val_scaled = np.concatenate(
                    (
                        self.val_scaled,
                        self.dtf_val[["hour", "day", "weekday", "month"]].values,
                    ),
                    axis=1,
                )
                self.test_scaled = np.concatenate(
                    (
                        self.test_scaled,
                        self.dtf_test[["hour", "day", "weekday", "month"]].values,
                    ),
                    axis=1,
                )
                self.test_small_scaled = np.concatenate(
                    (
                        self.test_small_scaled,
                        self.dtf_test_small[["hour", "day", "weekday", "month"]].values,
                    ),
                    axis=1,
                )

            self.scaler = scaler

            def to_ds(ds, step, raw_data):
                out_features_mask = None

                ds = ContextBatchDataset(
                    Features=torch.from_numpy(ds[:, self.bahnlast_mask]).to(
                        torch.float32
                    ),
                    Context=torch.from_numpy(ds[:, ~self.bahnlast_mask]).to(
                        torch.float32
                    ),
                    out_features_mask=out_features_mask,
                    ts_in=self.ts_in,
                    ts_out=self.ts_out,
                    step=step,
                    raw_data=raw_data,
                )
                return ds

            self.datasets = dict(
                train_test=to_ds(self.train_scaled, self.step_test, self.train),
                train=to_ds(self.train_scaled, self.step_train, self.train),
                val=to_ds(self.val_scaled, self.step_val, self.val),
                test=to_ds(self.test_scaled, self.step_test, self.test),
                test_small=to_ds(self.test_small_scaled, self.step_test, self.test),
            )

    def inverse_scale_bahnlast(self, X):
        x = np.zeros((X.shape[0], self.train_scaled.shape[1]))
        x[:, self.bahnlast_mask] = X

        if self.datetime_features:
            res = self.scaler.inverse_transform(x[:, self.scaler_mask])
        else:
            res = self.scaler.inverse_transform(x)

        return res[:, self.bahnlast_mask_orig]

    def _create_dataloader(self, ds, batch_size, shuffle=True):
        sampler = BatchSampler(
            RandomSampler(ds) if shuffle else SequentialSampler(ds),
            batch_size=batch_size,
            drop_last=self.drop_last,
        )
        loader_args = dict(
            sampler=sampler,
            num_workers=self.num_workers,
            batch_size=None,  # otherwise it adds a dimension with size 1 x ...
        )
        return DataLoader(ds, **loader_args)

    def to_sklearn(self, dl, lag):
        return dl_to_sklearn__with_context_for_tabpfn(dl=dl, lag=lag)

    def train_dataloader(self):
        return self._create_dataloader(
            self.datasets["train"], batch_size=self.batch_size
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.datasets["val"], batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return self._create_dataloader(
            self.datasets["test"], batch_size=self.batch_size, shuffle=False
        )

    def test_small_dataloader(self):
        return self._create_dataloader(
            self.datasets["test_small"], batch_size=self.batch_size, shuffle=False
        )
