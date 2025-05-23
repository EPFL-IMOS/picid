import pytest
import hydra
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra

class TestTimeSeriesDatasetLoaders:
    """Test suite for loading and validating time series datasets."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup paths and config list."""
        GlobalHydra.instance().clear()
        self.base_dir = Path(__file__).resolve().parents[3]
        self.relative_config_dir = "../../../configs"
        self.config_names = ["ETTh1_refactored", "railway"]# "cmapss_refactored"
        self.test_splits = ['train', 'val', 'test']

    def _load_dataset(self, config_name: str):
        """Hydra-based dataset instantiation."""
        with hydra.initialize(version_base="1.3", config_path=self.relative_config_dir):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[f"dataset={config_name}"],
                return_hydra_config=True,
            )
            dataset = hydra.utils.instantiate(cfg.dataset.loader)
            return dataset

    def test_loader_behavior(self):
        """Test that each dataset is properly loaded and processed."""
        for config_name in self.config_names:
            dataset = self._load_dataset(config_name)

            # Validate basic functionality
            features, timestamps = dataset.get_data()
            scaler = dataset.get_scaler()

            for split in self.test_splits:
                assert split in features, f"{split} not in features for {config_name}"
                assert split in timestamps, f"{split} not in timestamps for {config_name}"
                assert features[split].shape[0] == timestamps[split].shape[0], \
                    f"Mismatch in {split} lengths for {config_name}"

            if hasattr(scaler, "transform"):
                transformed = scaler.transform(features["train"])
                assert transformed.shape == features["train"].shape, \
                    f"Scaler output shape mismatch for {config_name}"

            print(f"✅ {config_name} dataset loaded and validated.")


    
       


    
                
