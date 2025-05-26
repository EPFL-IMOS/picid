import pytest
import hydra
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra


from lmetk.data.datasets import SWDataset, ContextBatchDataset

from lmetk.data.utils import reconstruct_datetime_from_features


class TestRailwayDatasets:
    """Test suite for loading and validating time series datasets."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup paths and config list."""
        GlobalHydra.instance().clear()
        self.base_dir = Path(__file__).resolve().parents[3]
        self.relative_config_dir = "../../../configs"
        self.config_names = ["railway"]
        self.test_splits = ["test"]

    def _load_dataset(self, config_name: str):
        """Hydra-based dataset instantiation."""
        with hydra.initialize(version_base="1.3", config_path=self.relative_config_dir):
            self.cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[f"dataset={config_name}"],
                return_hydra_config=True,
            )
            dataset = hydra.utils.instantiate(self.cfg.dataset.loader)
            dataset._pipeline()

            return dataset

    def test_timeseries_dataset(self):
        """Test that each dataset is properly loaded and processed."""
        for config_name in self.config_names:
            dataset = self._load_dataset(config_name)

            # Validate basic functionality
            splitted_data, splitted_timestamps = dataset.get_data()

        data_splitter_params = self.cfg.dataset.loader.data_splitter
        seq_len = data_splitter_params.seq_len
        label_len = data_splitter_params.label_len
        pred_len = data_splitter_params.pred_len

        dataset_test = SWDataset(
            features=splitted_data["test"],
            time_features=splitted_timestamps["test"],
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
        )

        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset_test.__getitem__(seq_len)
        assert (
            str(
                reconstruct_datetime_from_features(
                    seq_x_mark, ["year", "month", "day", "weekday", "hour"]
                )[0]
            )
            == "2023-01-24 00:00:00"
        ), "Timeseries dataset is not starting from the right daytime."

        dataset_test = ContextBatchDataset(
            features=splitted_data["test"],
            time_features=splitted_timestamps["test"],
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            mask=dataset.data_dict["mask"],
            stride=1,
        )

        d = dataset_test.__getitem__(seq_len)
        assert (
            str(
                reconstruct_datetime_from_features(
                    d["target"]["x_time"].long().numpy(),
                    ["year", "month", "day", "weekday", "hour"],
                )[0]
            )
            == "2023-01-24 00:00:00"
        ), "Datetime feature reconstruction error"

        print(f"✅ {config_name} dataset loaded and validated.")
