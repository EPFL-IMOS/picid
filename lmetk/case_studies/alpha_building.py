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


class BuildingEnergyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        test_ratio: float = 0.5,
        val_ratio: float = 0.084,
        train_ratio: float = 1.0,
        train_scaled: Optional[None] = None,
        val_scaled: Optional[None] = None,
        test_scaled: Optional[None] = None,
        batch_size=64,
        samples_per_day=None,
        trend_decomp_rolling_window=384,
        ts_in=336,
        ts_out=48,
        step_train=1,
        step_val=48,
        step_test=48,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        data_split_strategy="default",
        include_bl_forecast=False,
        datetime_features=True,
        forecast_feature_name="",
        forecast_all_features=False,
        trend_decomp="disabled",
        subsample_strategy=None,
        spatial_aggregation: Optional[str] = None,
    ):
        super().__init__()
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = train_ratio

        if train_ratio < 1.0:
            assert data_split_strategy == "scale_training_set", (
                "Only scale_training_set strategy is supported"
            )

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
        self.data_dir = data_dir
        self.spatial_aggregation = spatial_aggregation

    def load_raw_data(self, data_dir):
        data_path = data_dir
        data = pd.read_csv(data_path, sep=",")
        data.set_index("timestamp", inplace=True)

        return data

    def setup(self, stage: Optional[str] = None):
        if self.train_scaled is None:
            data = self.load_raw_data(self.data_dir)

            # use this to downsample the data
            data_samples_per_day = (
                data.index.to_series().groupby(data.index.date).count().iloc[0]
            )
            samples_per_day = data_samples_per_day

            assert self.samples_per_day is None, "Subsampling not implemented"
            assert self.subsample_strategy is None, "Not implemented"

            if not (
                self.subsample_strategy == "mean_1h" or self.subsample_strategy == None
            ):
                raise ValueError(
                    "Only mean_1h is supported, make sure the dataset is aligned with the small one"
                )

            if self.subsample_strategy == "select_1h":
                hourly = pd.Series(
                    data=pd.date_range(
                        start=data.index.min(), end=data.index.max(), freq="1h"
                    )
                )
                data = data.loc[hourly]
            if self.subsample_strategy == "mean_1h":
                data["Bahnlast"] = data["Bahnlast"].shift(-2).ffill()
                data = data.shift(1).bfill().resample("1h").mean()
                data = data.resample("1H").mean()

            self.data = {"full": data}

            """
            WARNING: comumns need to be in order:
            Bahnlast <forecast targets>, covaraites <contexts> ...
            The unintuitive part is that if there are N variables in the context
            sequence (the encoder input) and K (K < N) variables in the target sequence
            (the decoder input), then this implementation of Local Cross Attn
            assumes that the first K variables in the context correspond to the
            first K in the target. This means that if the context sequence is shape 
            (batch, length, N), then context[:, :, :K] gets you the context of the
            K target variables (target[..., i] is the same variable
            as context[..., i]). If this isn't true the model will still train but
            you will be connecting variables by cross attention in a very arbitrary
            way. Note that the built-in CSVDataset *does* account for this and always
            puts the target variables in the same order in the lowest indices of both
            sequences. ** If your target variables do not appear in the context sequence
            Local Cross Attention should almost definitely be turned off
            (--local_cross_attn none) **.
            """

            if self.spatial_aggregation is not None:
                print("Spatial aggregation")

            if self.trend_decomp == "disabled":
                gt = data[
                    [
                        "Site Total Energy (kWh)",
                    ]
                ]
                self.data = {"full": data, "gt": gt.copy()}
            else:
                raise ValueError("Unknown trend decomposition")

            if not self.forecast_all_features:
                # self.features_mask = (data.columns == "Bahnlast")
                self.bahnlast_mask = data.columns == self.forecast_feature_name
                self.bahnlast_mask_orig = data.columns == self.forecast_feature_name
            else:
                self.bahnlast_mask = np.ones(data.shape[1], dtype=bool)
                self.bahnlast_mask_orig = np.ones(data.shape[1], dtype=bool)

            assert check_boolean_vector(self.bahnlast_mask), (
                "Should be boolean vector in order 1111...000"
            )
            assert check_boolean_vector(self.bahnlast_mask_orig), (
                "Should be boolean vector in order 1111...000"
            )

            test_ratio = self.test_ratio
            val_ratio = self.val_ratio
            train_ratio = self.train_ratio

            cut_to_days = lambda x: (x // samples_per_day) * samples_per_day
            N_samples = cut_to_days(len(data))
            if self.data_split_strategy == "default":
                # if you modify this: this is very fragile to keep the DS day aligned!
                N_test = cut_to_days(round(test_ratio * N_samples))
                N_val = cut_to_days(round(val_ratio * N_samples))
                N_train = cut_to_days(N_samples - N_test - N_val)
                print(N_train)

                r_train = range(0, N_train)
                r_val = range(N_train, N_train + N_val)
                r_test = range(N_train + N_val, N_train + N_val + N_test)

            print("--" * 40)
            print("Dataset Split Summary:")
            print(
                f"Train: {len(r_train)} , Start Date: {data.index[r_train[0]]} , End Date: {data.index[r_train[-1]]}"
            )
            print(
                f"Val: {len(r_val)} , Start Date: {data.index[r_val[0]]} , End Date: {data.index[r_val[-1]]}"
            )
            print(
                f"Test: {len(r_test)} , Start Date: {data.index[r_test[0]]} , End Date: {data.index[r_test[-1]]}"
            )
            print("--" * 40)

            self.r_train, self.r_val, self.r_test = r_train, r_val, r_test
            self.train = data.iloc[r_train, :]
            self.val = data.iloc[r_val, :]
            self.test = data.iloc[r_test, :]

            scaler = MinMaxScaler()
            scaler.fit(self.train)
            self.train_scaled = scaler.transform(self.train)
            self.val_scaled = scaler.transform(self.val)
            self.test_scaled = scaler.transform(self.test)

            if self.datetime_features:
                dtf = time_features(
                    pd.Series(data.index),
                    use_features=["month", "day", "weekday", "hour"],
                )
                self.dtf_train = dtf.iloc[r_train, :]
                self.dtf_val = dtf.iloc[r_val, :]
                self.dtf_test = dtf.iloc[r_test, :]

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

            self.scaler = scaler

            def to_ds(ds, step, raw_data):
                # out_features_mask = torch.from_numpy(data.columns == self.forecast_feature_name).to(torch.bool) \
                #     if self.forecast_all_features else None

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
            # persistent_workers=True,
            batch_size=None,  # otherwise it adds a dimension with size 1 x ...
            # due to the batch sampler providing the batch dimension already
        )
        return DataLoader(ds, **loader_args)

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
