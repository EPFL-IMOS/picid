from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def check_boolean_vector(vec):
    # Find the index of the first False value
    try:
        first_false_index = np.where(vec == False)[0][0]
    except IndexError:  # If there is no False in the vector
        return True

    # Check if all values after the first False are also False
    return not vec[first_false_index:].any() and vec[:first_false_index].all()


def time_features(
    dates: pd.Series,
    main_df=None,
    use_features=["year", "month", "day", "weekday", "hour", "minute"],
    time_col_name="Datetime",
):
    if main_df is None:
        main_df = pd.DataFrame({})
    else:
        main_df = main_df.copy()
    years = dates.apply(lambda row: row.year)
    max_year = years.max()
    min_year = years.min()

    norm_f = lambda r: 2.0 * r - 1.0
    norm_f = lambda r: r

    if "year" in use_features:
        main_df["year"] = dates.apply(
            lambda row: (row.year - min_year) / max(1.0, (max_year - min_year))
        )

    if "month" in use_features:
        main_df["month"] = dates.apply(lambda row: norm_f((row.month - 1) / 11.0))
    if "day" in use_features:
        main_df["day"] = dates.apply(lambda row: norm_f((row.day - 1) / 30.0))

    if "weekday" in use_features:
        main_df["weekday"] = dates.apply(lambda row: norm_f(row.weekday() / 6.0))
    if "hour" in use_features:
        main_df["hour"] = dates.apply(lambda row: norm_f((row.hour) / 23.0))

    if "minute" in use_features:
        main_df["minute"] = dates.apply(lambda row: ((row.minute) / 59.0))

    main_df[time_col_name] = dates
    return main_df


class IndexSequencer:
    def __init__(
        self,
        F: torch.Tensor,
        ts_in: int,
        ts_out: int,
        dt_ts_out: List[int] = None,
        step: int = 1,
    ) -> None:
        self.F = F

        if dt_ts_out is not None:
            dt_ts_out.sort()
            dt_ts_out = np.array(dt_ts_out)
            dt_max = dt_ts_out[-1]
            upper_limit = max(ts_out, dt_max + 1)
        else:
            upper_limit = ts_out

        N_samples = F.shape[0]

        i = torch.arange(0, N_samples - (ts_in + upper_limit) + 1, step=step)
        self.start_end_index = torch.stack([i, i + (ts_in + upper_limit)], dim=1)

        self.N_sequences = len(self.start_end_index)

    def __len__(self):
        return self.N_sequences

    def sequences_batch(
        self, index_subset: torch.Tensor | np.ndarray
    ) -> Tuple[torch.Tensor]:
        sel = self.start_end_index[index_subset]
        return self._create_sequences(sel)

    @abstractmethod
    def _create_sequences(self, sel: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        pass


class AutoregressiveSequencer(IndexSequencer):
    def __init__(
        self,
        F: torch.Tensor,
        ts_in: int,
        ts_out: int,
        T: torch.Tensor = None,
        in_features_mask=None,
        out_features_mask=None,
        step: int = 10,
    ) -> None:
        super().__init__(F=F, ts_in=ts_in, ts_out=ts_out, step=step)
        self.ts_in = ts_in
        self.ts_out = ts_out
        self.in_features_mask = in_features_mask
        self.out_features_mask = out_features_mask
        self.T = F

    def _create_sequences(self, sel: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        ts_in, ts_out = self.ts_in, self.ts_out
        F, T = self.F, self.T

        f_unfolded, t_unfolded = [], []
        for i, j in sel:
            f_unfolded.append(F[i : i + ts_in, ...])
            t_unfolded.append(T[i + ts_in : j, ...])

        f_unfolded = torch.stack(f_unfolded, dim=0)
        t_unfolded = torch.stack(t_unfolded, dim=0)

        if self.in_features_mask is not None:
            f_unfolded = f_unfolded[..., self.in_features_mask]

        if self.out_features_mask is not None:
            t_unfolded = t_unfolded[..., self.out_features_mask]

        return f_unfolded, t_unfolded


class ContextBatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        Features: torch.Tensor,
        Context: torch.Tensor,
        # sequence length
        ts_in: int,  # label length
        ts_out: int,  # prediction horizon
        out_features_mask: Optional[torch.Tensor] = None,  # not needed
        step: int = 1,  # strite over sequences
        raw_data: Optional[pd.DataFrame] = None,
    ):
        self.raw_data = raw_data
        self.F_seq = AutoregressiveSequencer(
            Features, ts_in, ts_out, step=step, out_features_mask=out_features_mask
        )
        self.C_seq = AutoregressiveSequencer(Context, ts_in, ts_out, step=step)

        self.ts_in = ts_in

    def get_raw_data(self, batch_idx: List[int]):
        if self.raw_data is not None:
            ts_in = self.ts_in
            sel = self.F_seq.start_end_index[batch_idx]

            f_unfolded, t_unfolded = [], []
            for i, j in sel:
                i, j = i.item(), j.item()
                f_unfolded.append(self.raw_data.iloc[i : i + ts_in, ...])
                t_unfolded.append(self.raw_data.iloc[i + ts_in : j, ...])

            return f_unfolded, t_unfolded
        else:
            raise ValueError("No raw data available")

    def __len__(self):
        return len(self.F_seq)

    def __getitem__(self, batch_idx: List[int]):
        F_f, F_t = self.F_seq.sequences_batch(batch_idx)
        C_f, C_t = self.C_seq.sequences_batch(batch_idx)
        return C_f, F_f, C_t, F_t
