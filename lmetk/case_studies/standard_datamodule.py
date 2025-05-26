from typing import Optional
import lightning as pl
from omegaconf import OmegaConf
import torch
from einops import rearrange

from lmetk.case_studies.tabularizer import HistoryAndContextTabularizer
from .data_factory import data_provider


class StandardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        batch_size=32,
        ts_in=24 * 4,
        ts_out=24 * 4,
        step_train=24 * 4,
        step_val=24 * 4,
        step_test=24 * 4,
        shuffle=True,
        num_workers=0,
        start_token_len=0,
        datetime_features=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.ts_in = ts_in
        self.ts_out = ts_out
        self.start_token_len = start_token_len

        self.step_train = step_train
        self.step_val = step_val
        self.step_test = step_test
        self.shuffle = shuffle

        self.datetime_features = datetime_features
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.dataset_name = dataset_name

    def setup(self, stage: Optional[str] = None):
        if self.dataset_name == "ETTh1":
            data = "ETTh1"
            data_path = "ETTh1.csv"
        elif self.dataset_name == "ETTh2":
            data = "ETTh2"
            data_path = "ETTh2.csv"

        args = OmegaConf.create(
            dict(
                freq="h",
                batch_size=self.batch_size,
                task_name="forecasting",
                seasonal_patterns="Monthly",
                data=data,
                embed="timeF",
                augmentation_ratio=0.0,
                root_path=self.data_dir,
                data_path=data_path,
                seq_len=self.ts_in,  # context sequence
                label_len=self.start_token_len,  # start token (decoder)
                pred_len=self.ts_out,  # prediction sequence
                features="M",
                target="OT",
                num_workers=self.num_workers,
                scale=True,
            )
        )

        # start token length changes the dataset size ??
        assert self.start_token_len == 0, "Start token length must be zero for now"

        self.ds_train, self.dl_train = data_provider(args, flag="train")
        self.ds_test, self.dl_test = data_provider(args, flag="test")
        self.ds_val, self.dl_val = data_provider(args, flag="val")

        self.datasets = {
            "train": self.ds_train,
            "test": self.ds_test,
            "val": self.ds_val,
        }

    def to_sklearn(self, dl, lag=0):
        tabularizer = HistoryAndContextTabularizer()

        def map_batch(batch):
            # original naming convention seq_x, seq_y, seq_x_mark, seq_y_mark = batch
            # our convention
            y_c, y_t, x_c, x_t = batch
            return x_c, y_c, x_t, y_t

        tabular_data = tabularizer.tabularize(dl, lag, map_batch_f=map_batch)
        return tabular_data["X"], tabular_data["y"]

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_val

    def test_dataloader(self):
        return self.dl_test
