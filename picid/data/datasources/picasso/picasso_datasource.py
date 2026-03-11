import os.path
from pathlib import Path
from typing import Optional, List, Dict
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from omegaconf import ListConfig
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import warnings

from picid.data.datasources.base.single_source_loader import SingleSourceLoader

warnings.filterwarnings("ignore")


class PicassoLoader(SingleSourceLoader):
    pass


# class ContextBatchDatasetEnhanced(Dataset):
#     """
#     Enhanced version compatible with existing training strategy
#     """

#     def __init__(self, Features, Context_past, Context_future,
#                  past_context_steps, future_context_steps,
#                  context_col_mapping_past, context_col_mapping_future,
#                  ts_in, ts_out, lags_targets, step, raw_data=None, out_features_mask=None):
#         self.Features = Features  # Target features (TNG_POS, TNG_NEG)
#         self.Context_past = Context_past
#         self.Context_future = Context_future
#         self.past_context_steps = past_context_steps
#         self.future_context_steps = future_context_steps
#         self.context_col_mapping_past = context_col_mapping_past
#         self.context_col_mapping_future = context_col_mapping_future
#         self.ts_in = ts_in
#         self.ts_out = ts_out
#         self.lags_targets = lags_targets
#         self.step = step
#         self.raw_data = raw_data
#         self.out_features_mask = out_features_mask

#     def __len__(self):
#         return len(self.Features) - self.ts_in - self.ts_out + 1

#     def __getitem__(self, idx):
#         # Handle case where idx might be a list/tensor (from BatchSampler)
#         if isinstance(idx, (list, torch.Tensor)):
#             # If idx is a batch, process each index separately
#             batch_data = []
#             for single_idx in idx:
#                 batch_data.append(self._get_single_item(single_idx))

#             # Stack all batch items
#             x_c = torch.stack([item[0] for item in batch_data])
#             y_c = torch.stack([item[1] for item in batch_data])
#             x_t = torch.stack([item[2] for item in batch_data])
#             y_t = torch.stack([item[3] for item in batch_data])

#             return x_c, y_c, x_t, y_t
#         else:
#             # Single index case
#             return self._get_single_item(idx)

#     def _get_single_item(self, idx):
#         """Get a single item by index"""
#         # Ensure idx is an integer
#         if isinstance(idx, torch.Tensor):
#             idx = idx.item()

#         # Past sequences
#         y_c = self.Features[idx:idx + self.ts_in - self.lags_targets]  # Past target features

#         # Process past context with different steps for each column
#         x_c_past_list = []
#         for col_name, steps in self.past_context_steps.items():
#             col_idx = self.context_col_mapping_past[col_name]  # Map column name to index
#             past_start = max(0, idx + self.ts_in - steps)  # Ensure we don't go negative
#             past_end = idx + self.ts_in

#             col_data = self.Context_past[past_start:past_end, col_idx:col_idx + 1]

#             # Pad if we don't have enough historical data
#             if col_data.shape[0] < steps:
#                 pad_size = steps - col_data.shape[0]
#                 padding = torch.zeros((pad_size, 1))
#                 col_data = torch.cat([padding, col_data], dim=0)

#             x_c_past_list.append(col_data.flatten())

#         # Future sequences
#         future_start = idx + self.ts_in

#         # Process future context with different steps for each column
#         x_c_future_list = []
#         for col_name, steps in self.future_context_steps.items():
#             col_idx = self.context_col_mapping_future[col_name]
#             future_end = future_start + steps

#             # Select appropriate future context source based on horizon
#             col_data = self.Context_future[future_start:future_end, col_idx:col_idx + 1]

#             # Pad if needed
#             if col_data.shape[0] < steps:
#                 pad_size = steps - col_data.shape[0]
#                 padding = torch.zeros((pad_size, 1))
#                 col_data = torch.cat([col_data, padding], dim=0)

#             x_c_future_list.append(col_data.flatten())

#         # Concatenate all context features
#         if x_c_future_list or x_c_past_list:
#             x_c = torch.cat([
#                 *x_c_past_list,
#                 *x_c_future_list
#             ])
#         else:
#             x_c = torch.empty(0, dtype=torch.float32)

#         # Target sequences
#         y_t = self.Features[future_start:future_start+self.ts_out]

#         # For compatibility with existing training strategy
#         # Use appropriate context for x_t
#         if hasattr(self, 'Context_future') and self.Context_future.shape[0] > future_start+self.ts_out:
#             x_t = self.Context_future[future_start:future_start+self.ts_out]
#         else:
#             # Create placeholder with correct dimensions
#             x_t = torch.zeros((self.ts_out, self.Context_future.shape[1]))

#         return x_c, y_c, x_t, y_t


class EnhancedCSVLongTermForecastingModule(pl.LightningDataModule):
    """
    Enhanced Data Module with specific feature availability constraints

    Feature Availability Rules:
    - id_index: Latest 4 values NOT available for prediction
    - TNG_POS, TNG_NEG: Latest 3 values NOT available for prediction

    Feature Categories:
    - Past only: pv_generation, wind_generation, total_generation, grid_load, total_load, id_index, TNG
    - Future 24h: forecasted_da_pv_generation, forecasted_da_total_generation, forecasted_grid_load, forecasted_total_load, temperature
    - Future 6h: day_ahead_price, FCR_price, aFRR_cap_price_pos, aFRR_cap_price_neg
    - Targets: TNG_POS, TNG_NEG
    """

    def __init__(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        step_train: int,
        step_val: int,
        step_test: int,
        data_dir: str,
        ts_in: int,
        ts_out: int,
        target_cols: List[str] = ["TNG_POS", "TNG_NEG"],
        datetime_features: bool = False,
        samples_per_day: Optional[int] = None,
        drop_cols: Optional[List] = None,
        batch_size: int = 32,
        lags_id_index: int = 3,
        lags_afrr_pos_activated: int = 3,
        lags_afrr_neg_activated: int = 3,
        lags_tng_pos: int = 3,
        lags_tng_neg: int = 3,
        lags_targets: int = 2,
        fill_na: bool = True,
        scaler_type: str = "standard",
        forecast_steps: int = 4,
        past_context_steps: dict = None,
        future_context_steps: dict = None,
        data_split_strategy: str = "default",
        halve_temp: float = None,
        select_days: bool = False,
        clip_value: float = None,
        pos_days: list = None,
        neg_days: list = None,
        exp_type: str = "regression",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Data splitting ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Steps for different splits
        self.step_train = step_train
        self.step_val = step_val
        self.step_test = step_test

        # Time series parameters
        self.ts_in = ts_in
        self.ts_out = ts_out

        self.forecast_steps = forecast_steps

        # DataLoader parameters
        self.drop_last = False
        self.num_workers = 0

        # Feature parameters
        self.datetime_features = datetime_features
        self.target_cols = (
            target_cols
            if isinstance(target_cols, (list, ListConfig))
            else [target_cols]
        )
        self.drop_cols = (
            drop_cols
            if isinstance(drop_cols, (list, ListConfig))
            else ([drop_cols] if drop_cols else [])
        )
        self.samples_per_day = samples_per_day

        # Lag parameters for availability constraints
        self.lags_id_index = lags_id_index  # id_index: latest 4 not available
        self.lags_afrr_pos_activated = lags_afrr_pos_activated
        self.lags_afrr_neg_activated = lags_afrr_neg_activated
        self.lags_tng_pos = lags_tng_pos
        self.lags_tng_neg = lags_tng_neg
        self.lags_targets = lags_targets  # TNG_POS/NEG: latest 3 not available

        # Processing parameters
        self.fill_na = fill_na
        self.scaler_type = scaler_type

        # Define feature categories
        past_context_steps = (
            dict(past_context_steps) if past_context_steps is not None else {}
        )
        self.past_context_steps = past_context_steps

        future_context_steps = (
            dict(future_context_steps) if future_context_steps is not None else {}
        )
        self.future_context_steps = future_context_steps

        self.context_past_columns = list(past_context_steps.keys())
        self.context_future_columns = list(future_context_steps.keys())

        # Define data split strategy
        self.data_split_strategy = data_split_strategy

        # Temperary feature, split test set to two equal parts
        self.halve_temp = halve_temp
        self.select_days = select_days
        self.pos_days = pos_days
        self.neg_days = neg_days

        # Capping value
        self.clip_value = clip_value
        self.exp_type = exp_type

    def setup(self, stage: Optional[str] = None):
        """Setup data loading and preprocessing"""

        # Load data
        dir_path = os.path.dirname(__file__)
        dd = Path(os.path.join(dir_path, self.data_dir))
        if dd.suffix == ".csv":
            data = pd.read_csv(self.data_dir)
            if "Unnamed: 0" in data.columns:
                data.rename(columns={"Unnamed: 0": "time"}, inplace=True)
            if "time" in data.columns:
                data["time"] = pd.to_datetime(data["time"])
                data.set_index("time", inplace=True)
        elif dd.suffix == ".xlsx":
            data = pd.read_excel(self.data_dir)
        else:
            raise ValueError(f"Unsupported file format: {dd.suffix}")

        # Drop specified columns
        if self.drop_cols:
            existing_drop_cols = [col for col in self.drop_cols if col in data.columns]
            if existing_drop_cols:
                data.drop(existing_drop_cols, axis=1, inplace=True)

        # Handle missing values
        if self.fill_na:
            # data.fillna(method='ffill', inplace=True)
            # data.fillna(method='bfill', inplace=True)
            data.fillna(0, inplace=True)

        # Apply availability constraints
        data = self._apply_availability_constraints(data)

        # Clip outliers
        if self.exp_type == "regression":
            data = self._apply_clipping(data)

        # Store original data
        self.data = data

        # Determine samples per day
        if self.samples_per_day is None:
            data_samples_per_day = (
                data.index.to_series().groupby(data.index.date).count().iloc[0]
            )
            samples_per_day = data_samples_per_day
        else:
            samples_per_day = self.samples_per_day

        # Split data to train/val/test
        self._split_data(data, samples_per_day)

        # Setup scalers and feature masks
        self._setup_features_and_scaling()

        # Create datasets
        self._create_datasets()

    def _apply_clipping(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply upperbound clipping on extreme values"""

        print(" Applying clipping")

        target_cols = self.target_cols
        for col in target_cols:
            if col not in data.columns:
                raise ValueError(f"Target column {col} is not present in input data")
            else:
                data[col] = data[col].clip(
                    lower=self.clip_value * (-1), upper=self.clip_value
                )

        return data

    def _apply_availability_constraints(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply availability constraints to features"""

        print(" Applying availability constraints...")

        # id_index: latest 4 values not available
        if "id_index" in data.columns and "id_index" not in self.target_cols:
            data["id_index"] = data["id_index"].shift(self.lags_id_index, fill_value=0)
            print(f"   id_index: shifted by {self.lags_id_index} timesteps")
        else:
            print("   id_index: not shifted because it's in target columns")

        if (
            "afrr_neg_activated" in data.columns
            and "afrr_neg_activated" not in self.target_cols
        ):
            data["afrr_neg_activated"] = data["afrr_neg_activated"].shift(
                self.lags_afrr_neg_activated, fill_value=0
            )
            print(
                f"   afrr_neg_activated: lags_afrr_neg_activated by {self.lags_afrr_neg_activated} timesteps"
            )
        else:
            print("   afrr_neg_activated: not shifted because it's in target columns")

        if (
            "afrr_pos_activated" in data.columns
            and "afrr_pos_activated" not in self.target_cols
        ):
            data["afrr_pos_activated"] = data["afrr_pos_activated"].shift(
                self.lags_afrr_pos_activated, fill_value=0
            )
            print(
                f"   afrr_pos_activated: afrr_pos_activated by {self.lags_afrr_pos_activated} timesteps"
            )
        else:
            print("   afrr_pos_activated: not shifted because it's in target columns")

        if "TNG_POS" in data.columns and "TNG_POS" not in self.target_cols:
            data["TNG_POS"] = data["TNG_POS"].shift(self.lags_tng_pos, fill_value=0)
            print(f"   TNG_POS: shifted by {self.lags_tng_pos} timesteps")
        else:
            print("   TNG_POS: not shifted because it's in target columns")

        if "TNG_NEG" in data.columns and "TNG_NEG" not in self.target_cols:
            data["TNG_NEG"] = data["TNG_NEG"].shift(self.lags_tng_neg, fill_value=0)
            print(f"   TNG_NEG: shifted by {self.lags_tng_neg} timesteps")
        else:
            print("   TNG_NEG: not shifted because it's in target columns")

        print(" Availability constraints applied")
        return data

    def _split_data(self, data: pd.DataFrame, samples_per_day: int):
        """Split data into train/val/test"""

        def _cut_to_days(x):
            return (x // samples_per_day) * samples_per_day

        N_samples = _cut_to_days(len(data))

        if self.data_split_strategy == "default":
            N_test = _cut_to_days(round(self.test_ratio * N_samples))
            N_val = _cut_to_days(round(self.val_ratio * N_samples))
            N_train = _cut_to_days(N_samples - N_test - N_val)

        elif self.data_split_strategy == "yearly":
            years = data.index.year.unique()
            N_train_val = len(data[data.index.year == years[0]])
            N_train = _cut_to_days(round(self.train_ratio * N_train_val))
            N_val = _cut_to_days(N_train_val - N_train)
            N_test = _cut_to_days(len(data) - N_train - N_val)

            if N_train <= 1e4:
                raise ValueError("Too little training data")

        r_train = range(0, N_train)
        r_val = range(N_train, N_train + N_val)
        r_test = range(N_train + N_val, N_train + N_val + N_test)

        self.r_train, self.r_val, self.r_test = r_train, r_val, r_test
        self.train = data.iloc[r_train, :]
        self.val = data.iloc[r_val, :]
        self.test = data.iloc[r_test, :]

        if (self.data_split_strategy == "yearly") and (self.halve_temp is not None):
            idx, last_idx = (int(i) for i in self.halve_temp.split("/"))
            idx_start = round(len(r_test) * (idx - 1) / last_idx)
            idx_end = round(len(r_test) * idx / last_idx)
            test_section = range(N_train + N_val + idx_start, N_train + N_val + idx_end)
            self.test = data.iloc[test_section, :]

        print("   Data split completed:")
        print(f"   Train: {len(self.train):,} samples")
        print(f"   Validation: {len(self.val):,} samples")
        print(f"   Test: {len(self.test):,} samples")

    def _setup_features_and_scaling(self):
        """Setup feature masks and scaling"""

        # Verify target columns exist
        targets_in_columns = self.data.columns.isin(self.target_cols)
        if targets_in_columns.sum() != len(self.target_cols):
            missing_targets = [
                col for col in self.target_cols if col not in self.data.columns
            ]
            raise ValueError(f"Missing target columns: {missing_targets}")

        self.target_cols_mask = targets_in_columns
        self.target_cols_mask_orig = targets_in_columns

        # Create feature masks for different categories
        self._create_feature_masks()

        # Choose scaler
        if self.scaler_type == "standard":
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif self.scaler_type == "minmax":
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Separate feature/target dataframes
        X_train = self.train.loc[:, ~targets_in_columns]
        y_train = self.train.loc[:, targets_in_columns]

        # Fit scaler on features
        scaler_X.fit(X_train)
        self.scaler_X = scaler_X

        # Handle targets (categorical vs numeric)

        y_train_scaled = y_train.copy()
        if self.exp_type == "regression":
            # numeric → scale
            scaler_y.fit(y_train[y_train.columns])
            self.scaler_y = scaler_y
            y_train_scaled = scaler_y.transform(y_train[y_train.columns])
            print(scaler_y.feature_names_in_)
        elif self.exp_type == "classification":
            # categorical → encode
            self.label_encoders_y = {}  # store encoders if categorical
            for col in y_train.columns:
                le = LabelEncoder()
                y_train_scaled[col] = le.fit_transform(y_train[col])
                self.label_encoders_y[col] = le
            # No scaler for classification targets
            self.scaler_y = None

        # Scale features for train/val/test
        def scale_dataset(df):
            X_scaled = scaler_X.transform(df.loc[:, ~targets_in_columns])
            X_scaled = pd.DataFrame(
                X_scaled, columns=df.loc[:, ~targets_in_columns].columns, index=df.index
            )

            y_part = df.loc[:, targets_in_columns].copy()

            if self.exp_type == "regression" and self.scaler_y is not None:
                # Scale numeric targets
                y_scaled = self.scaler_y.transform(y_part)
                y_part = pd.DataFrame(
                    y_scaled, columns=y_part.columns, index=y_part.index
                )
            elif self.exp_type == "classification" and hasattr(
                self, "label_encoders_y"
            ):
                # Encode categorical targets
                for col in y_part.columns:
                    if col in self.label_encoders_y:
                        y_part[col] = self.label_encoders_y[col].transform(y_part[col])

            print(
                f"X_scaled.columns={X_scaled.columns.tolist()}, y_part.columns={y_part.columns.tolist()}"
            )

            # Reconstruct the scaled dataset
            scaled_ds = df.copy()
            scaled_ds.loc[:, ~targets_in_columns] = X_scaled
            scaled_ds.loc[:, targets_in_columns] = y_part

            return scaled_ds

        train_scaled = scale_dataset(self.train)
        val_scaled = scale_dataset(self.val)
        test_scaled = scale_dataset(self.test)

        # Convert to numpy arrays
        self.train_scaled = train_scaled.values
        self.val_scaled = val_scaled.values
        self.test_scaled = test_scaled.values
        self.train_val_scaled = np.concatenate((self.train_scaled, self.val_scaled))

        # Add datetime features if requested
        if self.datetime_features:
            self._add_datetime_features()

    def _create_feature_masks(self):
        """Create masks for different feature categories"""

        all_columns = self.data.columns.tolist()

        # Create masks for each category
        self.context_past_mask = [
            col in self.context_past_columns for col in all_columns
        ]
        self.context_future_mask = [
            col in self.context_future_columns for col in all_columns
        ]

        # Create mapping from column names to indices
        context_past_cols = self.data.loc[:, self.context_past_mask].columns.tolist()
        self.context_col_mapping_past = {}
        for i in range(len(context_past_cols)):
            col_name = context_past_cols[i]
            self.context_col_mapping_past[col_name] = i

        context_future_cols = self.data.loc[
            :, self.context_future_mask
        ].columns.tolist()
        self.context_col_mapping_future = {}
        for i in range(len(context_future_cols)):
            col_name = context_future_cols[i]
            self.context_col_mapping_future[col_name] = i

        # Validate that all specified columns exist
        for col_name in list(self.past_context_steps.keys()) + list(
            self.future_context_steps.keys()
        ):
            if col_name not in context_past_cols + context_future_cols:
                raise ValueError(f"Column '{col_name}' not found in context_columns")

        self.context_past_mask = np.array(self.context_past_mask)
        self.context_future_mask = np.array(self.context_future_mask)

        print("  Feature categories:")
        print(f"   Past features: {sum(self.context_past_mask)} features")
        print(f"   Future features: {sum(self.context_future_mask)} features")
        print(f"   Targets: {sum(self.target_cols_mask)} features")

    def _add_datetime_features(self):
        """Add datetime features if requested"""

        from picid.case_studies.common import time_features

        dtf = time_features(
            pd.Series(self.data.index), use_features=["month", "day", "weekday", "hour"]
        )

        self.dtf_train = dtf.iloc[self.r_train, :]
        self.dtf_val = dtf.iloc[self.r_val, :]
        self.dtf_test = dtf.iloc[self.r_test, :]

        # Update masks
        self.scaler_mask = np.concatenate(
            (np.ones(self.train_scaled.shape[1], dtype=bool), np.zeros(4, dtype=bool))
        )

        self.target_cols_mask = np.concatenate(
            (self.target_cols_mask, np.zeros(4, dtype=bool))
        )
        self.context_future_mask = np.concatenate(
            (self.context_future_mask, np.ones(4, dtype=bool))
        )
        self.context_past_mask = np.concatenate(
            (self.context_past_mask, np.zeros(4, dtype=bool))
        )

        # Concatenate datetime features
        datetime_cols = ["hour", "day", "weekday", "month"]
        self.train_scaled = np.concatenate(
            (self.train_scaled, self.dtf_train[datetime_cols].values), axis=1
        )
        self.val_scaled = np.concatenate(
            (self.val_scaled, self.dtf_val[datetime_cols].values), axis=1
        )
        self.test_scaled = np.concatenate(
            (self.test_scaled, self.dtf_test[datetime_cols].values), axis=1
        )

        start_idx = len(set(self.context_col_mapping_future.keys()))
        # Update feature columns dictionary
        for i, datetime_type in enumerate(datetime_cols):
            self.future_context_steps[datetime_type] = 1
            self.context_col_mapping_future[datetime_type] = start_idx + i

    # def _create_datasets(self):
    #     """Create PyTorch datasets"""

    #     def to_ds(ds, step, raw_data):
    #         # Extract features by category
    #         target_features = torch.from_numpy(ds[:, self.target_cols_mask]).to(
    #             torch.float32
    #         )
    #         context_past = torch.from_numpy(ds[:, self.context_past_mask]).to(
    #             torch.float32
    #         )
    #         context_future = torch.from_numpy(ds[:, self.context_future_mask]).to(
    #             torch.float32
    #         )

    #         dataset = ContextBatchDatasetEnhanced(
    #             Features=target_features,
    #             Context_past=context_past,
    #             Context_future=context_future,
    #             past_context_steps=self.past_context_steps,
    #             future_context_steps=self.future_context_steps,
    #             context_col_mapping_past=self.context_col_mapping_past,
    #             context_col_mapping_future=self.context_col_mapping_future,
    #             ts_in=self.ts_in,
    #             ts_out=self.ts_out,
    #             lags_targets=self.lags_targets,
    #             step=step,
    #             raw_data=raw_data,
    #         )
    #         return dataset

    #     self.datasets = dict(
    #         train_val=to_ds(self.train_val_scaled, self.step_train, self.train),
    #         train=to_ds(self.train_scaled, self.step_train, self.train),
    #         val=to_ds(self.val_scaled, self.step_val, self.val),
    #         test=to_ds(self.test_scaled, self.step_test, self.test),
    #     )

    #     print(f"  Datasets created successfully")

    # def inverse_scale_targets(self, X):
    #     """Inverse scale target predictions"""
    #     x = np.zeros((X.shape[0], self.train_scaled.shape[1]))
    #     x[:, self.target_cols_mask] = X

    #     if self.datetime_features:
    #         res = self.scaler_y.inverse_transform(x[:, self.scaler_mask])
    #     else:
    #         res = self.scaler_y.inverse_transform(x)

    #     return res[:, self.target_cols_mask_orig]

    # def to_sklearn(self, dl, lag=0):
    #     """Convert DataLoader to sklearn format for compatibility"""
    #     X, y = [], []
    #     for batch in iter(dl):
    #         x_c, y_c, x_t, y_t = batch
    #         # Concatenate context and target features for input
    #         X.append(torch.concat([x_c.flatten(start_dim=1)], dim=1))
    #         # X.append(torch.concat([x_c.flatten(start_dim=1), y_c.flatten(start_dim=1)], dim=1)) #TODO: fix this later
    #         # X.append(torch.concat([x_c.flatten(start_dim=1), y_c.transpose(1, 2).reshape(y_c.size(0), -1)], dim=1))
    #         y.append(y_t[:, lag, :])

    #     return torch.concat(X).numpy(), torch.concat(y).numpy()

    # def _create_dataloader(self, ds, batch_size, shuffle=True):
    #     """Create DataLoader with proper sampling"""
    #     sampler = BatchSampler(
    #         RandomSampler(ds) if shuffle else SequentialSampler(ds),
    #         batch_size=batch_size,
    #         drop_last=self.drop_last,
    #     )

    #     loader_args = dict(
    #         sampler=sampler,
    #         num_workers=self.num_workers,
    #         batch_size=None,  # BatchSampler already provides batching
    #     )

    #     return DataLoader(ds, **loader_args)

    # def train_dataloader(self):
    #     """Create training DataLoader"""
    #     return self._create_dataloader(
    #         self.datasets["train"], batch_size=self.batch_size
    #     )

    # def val_dataloader(self):
    #     """Create validation DataLoader"""
    #     return self._create_dataloader(
    #         self.datasets["val"], batch_size=self.batch_size, shuffle=False
    #     )

    # def train_val_dataloader(self):
    #     """Create training-validating DataLoader"""
    #     return self._create_dataloader(
    #         self.datasets["train_val"], batch_size=self.batch_size
    #     )

    # def test_dataloader(self):
    #     """Create test DataLoader"""
    #     if self.select_days and (
    #         self.pos_days is not None or self.neg_days is not None
    #     ):
    #         subsets = []  # Temporary list to collect subsets

    #         has_pos = any("POS" in col for col in self.target_cols)
    #         has_neg = any("NEG" in col for col in self.target_cols)

    #         if has_pos:
    #             subsets.append(Subset(self.datasets["test"], self.pos_days))

    #         if has_neg:
    #             subsets.append(Subset(self.datasets["test"], self.neg_days))

    #         # Combine all subsets into a single dataset
    #         if len(subsets) == 1:
    #             subset_dataset = subsets[0]
    #         elif len(subsets) > 1:
    #             subset_dataset = ConcatDataset(subsets)
    #         else:
    #             # Fallback if no conditions matched
    #             subset_dataset = self.datasets["test"]

    #         return self._create_dataloader(
    #             subset_dataset, batch_size=self.batch_size, shuffle=False
    #         )

    #     return self._create_dataloader(
    #         self.datasets["test"], batch_size=self.batch_size, shuffle=False
    #     )

    def get_feature_info(self) -> Dict:
        """Get information about features and their categories"""
        return {
            "target_features": self.target_cols,
            "past_features": [
                col for col in self.data.columns if col in self.context_past_columns
            ],
            "future_features": [
                col for col in self.data.columns if col in self.context_future_columns
            ],
            "total_features": len(self.data.columns),
            "lags_applied": {
                "id_index": self.lags_id_index,
                "TNG_POS": self.lags_tng_pos,
                "TNG_NEG": self.lags_tng_neg,
                "targets": self.lags_targets,
            },
        }

    def print_data_summary(self):
        """Print comprehensive data summary"""
        feature_info = self.get_feature_info()

        print("\n" + "=" * 60)
        print("ENHANCED ENERGY TRADING DATA MODULE SUMMARY")
        print("=" * 60)

        print("\n  Dataset Information:")
        print(f"   Total samples: {len(self.data):,}")
        print(f"   Features: {feature_info['total_features']}")
        print(f"   Targets: {len(feature_info['target_features'])}")

        print("\n  Target Features:")
        for target in feature_info["target_features"]:
            print(f"   - {target}")

        print("\n  Feature Categories:")
        print(
            f"   Past only ({len(feature_info['past_only_features'])}): {feature_info['past_only_features']}"
        )
        print(
            f"   Future 24h ({len(feature_info['future_24h_features'])}): {feature_info['future_24h_features']}"
        )
        print(
            f"   Future 6h ({len(feature_info['future_6h_features'])}): {feature_info['future_6h_features']}"
        )

        print("\n  Availability Constraints:")
        print(
            f"   id_index: {feature_info['lags_applied']['id_index']} timesteps delay"
        )
        print(f"   Targets: {feature_info['lags_applied']['targets']} timesteps delay")

        print("\n  Data Splits:")
        print(f"   Train: {len(self.train):,} samples ({self.train_ratio:.1%})")
        print(f"   Validation: {len(self.val):,} samples ({self.val_ratio:.1%})")
        print(f"   Test: {len(self.test):,} samples ({self.test_ratio:.1%})")

        print("\n  Model Parameters:")
        print(f"   Input length (ts_in): {self.ts_in}")
        print(f"   Output length (ts_out): {self.ts_out}")
        print(f"   Batch size: {self.batch_size}")

        print("=" * 60)
