from typing import Optional
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch

from lmetk.case_studies.common import ContextBatchDataset, time_features

from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler


class CSVLongTermForecastingModule(pl.LightningDataModule):
    def __init__(self,
                 train_ratio: float,
                 val_ratio: float,
                 test_ratio: float,
                 step_train: int,
                 step_val: int,
                 step_test: int,
                 data_dir: str,
                 ts_in: int,
                 ts_out: int,
                 target_col: str,
                 drop_cols: Optional[list] = None,
                 batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.step_train = step_train
        self.step_val = step_val
        self.step_test = step_test

        self.ts_in = ts_in
        self.ts_out = ts_out

        self.drop_last = False
        self.num_workers = 0

        self.data_split_strategy = "default"
        self.datetime_features = False
        self.target_col = target_col
        self.drop_cols = drop_cols

    def setup(self, stage: Optional[str] = None):
        data = pd.read_csv(self.data_dir)
        data['time'] = pd.to_datetime(data.time)
        data.set_index('time', inplace=True)

        if self.drop_cols:
            data = data.drop(self.drop_cols, axis=1)

        self.data = data

        # use this to downsample the data
        data_samples_per_day = data.index.to_series().groupby(data.index.date).count().iloc[0]
        samples_per_day = data_samples_per_day

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

        self.r_train, self.r_val, self.r_test = r_train, r_val, r_test
        self.train = data.iloc[r_train, :]
        self.val = data.iloc[r_val, :]
        self.test = data.iloc[r_test, :]

        self.target_col_mask = (data.columns == self.target_col)
        self.target_col_mask_orig = (data.columns == self.target_col)

        scaler = MinMaxScaler()
        scaler.fit(self.train)
        self.train_scaled = scaler.transform(self.train)
        self.val_scaled = scaler.transform(self.val)
        self.test_scaled = scaler.transform(self.test)

        if self.datetime_features:
            dtf = time_features(
                pd.Series(data.index),
                use_features=["month", "day", "weekday", "hour"])
            self.dtf_train = dtf.iloc[r_train, :]
            self.dtf_val = dtf.iloc[r_val, :]
            self.dtf_test = dtf.iloc[r_test, :]

            self.scaler_mask = np.concatenate(
                (np.ones(self.train_scaled.shape[1], dtype=bool), np.zeros(4, dtype=bool)))

            self.target_col_mask = np.concatenate(
                (self.target_col_mask, np.zeros(4, dtype=bool)))

            self.train_scaled = np.concatenate(
                (self.train_scaled, self.dtf_train[['hour', 'day', 'weekday', 'month']].values), axis=1)
            self.val_scaled = np.concatenate(
                (self.val_scaled, self.dtf_val[['hour', 'day', 'weekday', 'month']].values), axis=1)
            self.test_scaled = np.concatenate(
                (self.test_scaled, self.dtf_test[['hour', 'day', 'weekday', 'month']].values), axis=1)
            # self.test_small_scaled = np.concatenate(
            #     (self.test_small_scaled, self.dtf_test_small[['hour', 'day', 'weekday', 'month']].values), axis=1)

        self.scaler = scaler

        def to_ds(ds, step, raw_data):

            out_features_mask = None

            ds = ContextBatchDataset(
                Features=torch.from_numpy(ds[:, self.target_col_mask]).to(torch.float32),
                Context=torch.from_numpy(ds[:, ~self.target_col_mask]).to(torch.float32),
                out_features_mask=out_features_mask,
                ts_in=self.ts_in,
                ts_out=self.ts_out,
                step=step,
                raw_data=raw_data
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
        x[:, self.target_col_mask] = X

        if self.datetime_features:
            res = self.scaler.inverse_transform(x[:, self.scaler_mask])
        else:
            res = self.scaler.inverse_transform(x)

        return res[:, self.target_col_mask_orig]

    def to_sklearn(self, dl, lag=0):
        X, y = [], []
        for batch in iter(dl):
            x_c, y_c, x_t, y_t = batch
            X.append(torch.concat([x_t[:, lag, :], y_c.squeeze(2)], dim=1))
            y.append(y_t[:, lag, :])

        return torch.concat(X).numpy(), torch.concat(y).numpy()

    def _create_dataloader(self, ds, batch_size, shuffle=True):

        sampler = BatchSampler(
            RandomSampler(ds) if shuffle else SequentialSampler(ds),
            batch_size=batch_size,
            drop_last=self.drop_last
        )
        loader_args = dict(
            sampler=sampler,
            num_workers=self.num_workers,
            # persistent_workers=True,
            batch_size=None  # otherwise it adds a dimension with size 1 x ...
            # due to the batch sampler providing the batch dimension already
        )
        return DataLoader(ds, **loader_args)

    def train_dataloader(self):
        return self._create_dataloader(self.datasets['train'], batch_size=self.batch_size)

    def val_dataloader(self):
        return self._create_dataloader(self.datasets['val'], batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.datasets['test'], batch_size=self.batch_size, shuffle=False)
