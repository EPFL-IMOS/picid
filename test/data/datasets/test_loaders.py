
import pytest

list_of_available_datasets = ["ETTh1", "railway"]
@pytest.mark.parametrize("config_name", list_of_available_datasets)
def test_loader_behavior(config_name, datasource_loader, test_splits):
    """Test that each dataset is properly loaded and processed."""
    
    # Load the dataset using the factory fixture
    dataset = datasource_loader(config_name)

    # Validate basic functionality
    features, timestamps = dataset.get_data()
    scaler = dataset.get_scaler()

    for split in test_splits:
        assert split in features, f"{split} not in features for {config_name}"
        assert split in timestamps, (
            f"{split} not in timestamps for {config_name}"
        )
        assert features[split].shape[0] == timestamps[split].shape[0], (
            f"Mismatch in {split} lengths for {config_name}"
        )

    if hasattr(scaler, "transform"):
        transformed = scaler.transform(features["train"])
        assert transformed.shape == features["train"].shape, (
            f"Scaler output shape mismatch for {config_name}"
        )

    print(f"✅ {config_name} dataset loaded and validated.")

