import pytest
import numpy as np
import awkward as ak
import torch
from unittest.mock import patch, MagicMock, create_autospec
from omegaconf import OmegaConf
from numpy.testing import assert_array_equal

# --- Import the class to be tested
# (Adjust this import path to match your project structure)
from lmetk.transforms.battery.autoencoder_transform import AutoencoderTransform
from lmetk.transforms.base_transforms.autoencoder_transform import (
    BaseAutoencoderTransform,
)

# --- IMPORT THE UTILITY NEEDED FOR THE FIX ---
from lmetk.utils.awkward_utils import ak_regularize_regular_axes


@pytest.fixture
def transform_fixture():
    """
    Mocks the BaseAutoencoderTransform.__init__ to avoid loading files,
    and provides a fully mocked AutoencoderTransform instance.
    """
    # 1. Patch the base class __init__ so it does nothing
    with patch.object(
        BaseAutoencoderTransform, "__init__", lambda *args, **kwargs: None
    ):

        # 2. Instantiate our transform
        transform = AutoencoderTransform(
            folder_name="dummy/path", key="features", batch_size=2
        )

        # 3. Manually set attributes that __init__ would have
        transform.key = "features"
        transform.batch_size = 2
        transform.device = "cpu"

        # 4. Create a mock config (OmegaConf)
        # These values are crucial for the RaggedArraySequencer
        mock_cfg_dict = {
            "dataset": {
                "dataset_cfg": {
                    "seq_len": 3,
                    "label_len": 1,
                    "pred_len": 1,
                    "stride": 2,
                }
            },
            "model": {},  # Mock model config
            "trainer": {},  # Mock trainer config
        }
        transform._cfg = OmegaConf.create(mock_cfg_dict)

        # 5. Create a mock model
        transform._model = MagicMock()
        transform._model.backbone = MagicMock()

        # We will define the encode return_value/side_effect inside the specific
        # test that uses it, as the output depends on batch size.
        transform._model.backbone.encode = MagicMock()
        transform._model.backbone.__class__.__name__ = "MockedBackbone"

        yield transform


# --- Tests for Helper Methods ---

# [... All helper tests from test_validate_input... to
#  test_reconstruct_ragged_array remain unchanged ...]


def test_validate_input_raises_missing_features(transform_fixture):
    t = transform_fixture
    data = {"target": np.array([1, 2, 3])}
    with pytest.raises(AssertionError, match="Features key is not in data"):
        t._validate_input_data(data)


def test_validate_input_raises_missing_target(transform_fixture):
    t = transform_fixture
    data = {"features": np.array([1, 2, 3])}
    with pytest.raises(AssertionError, match="Target is not in data"):
        t._validate_input_data(data)


def test_validate_input_raises_wrong_feature_dims(transform_fixture):
    t = transform_fixture
    data = {
        "features": np.array([[1, 2], [3, 4]]),  # 2D
        "target": np.array([[1], [2]]),
    }
    with pytest.raises(AssertionError, match="Features must be 3D"):
        t._validate_input_data(data)


def test_validate_input_success(transform_fixture):
    t = transform_fixture
    feat = np.ones((5, 10, 3))
    targ = np.ones((5, 2))
    # Note: Using ak.Array here to simulate real ragged input
    data = {"features": ak.Array(feat), "target": ak.Array(targ)}

    f_out, t_out = t._validate_input_data(data)
    # _validate_input_data returns the raw array, so we compare content
    assert_array_equal(ak.to_numpy(f_out), feat)
    assert_array_equal(ak.to_numpy(t_out), targ)


def test_process_target_batch_3d_valid(transform_fixture):
    t = transform_fixture
    # Batch of 2, seq_len 3, 2 features
    # All values in [:, :, 0] are the same per-sample
    y_batch = np.array(
        [
            [[10, 0.1], [10, 0.2], [10, 0.3]],  # Sample 1
            [[20, 1.1], [20, 1.2], [20, 1.3]],  # Sample 2
        ]
    )
    result = t._process_target_batch(y_batch)

    # Should return the last time step: (batch_size, n_features)
    expected = np.array([[10, 0.3], [20, 1.3]])
    assert_array_equal(result, expected)
    assert result.shape == (2, 2)


def test_process_target_batch_3d_invalid_assertion(transform_fixture):
    t = transform_fixture
    # Values in [:, :, 0] are NOT the same
    y_batch = np.array(
        [
            [[10, 0.1], [10, 0.2], [10, 0.3]],  # Sample 1 (OK)
            [[20, 1.1], [21, 1.2], [20, 1.3]],  # Sample 2 (Bad)
        ]
    )
    with pytest.raises(AssertionError, match="Target values within the sequence"):
        t._process_target_batch(y_batch)


def test_process_target_batch_2d_valid(transform_fixture):
    t = transform_fixture
    # Batch of 2, seq_len 3. (e.g., from a 1D target)
    y_batch = np.array(
        [
            [100, 101, 102],  # Sample 1
            [200, 201, 202],  # Sample 2
        ]
    )
    result = t._process_target_batch(y_batch)

    # Should return the last time step: (batch_size,)
    expected = np.array([102, 202])
    assert_array_equal(result, expected)
    assert result.shape == (2,)


def test_process_target_batch_invalid_dims(transform_fixture):
    t = transform_fixture
    y_batch_1d = np.array([1, 2, 3])
    y_batch_4d = np.ones((2, 3, 4, 1))

    with pytest.raises(ValueError, match="Target batch has incorrect dimensions"):
        t._process_target_batch(y_batch_1d)

    with pytest.raises(ValueError, match="Target batch has incorrect dimensions"):
        t._process_target_batch(y_batch_4d)


def test_reconstruct_ragged_array(transform_fixture):
    t = transform_fixture
    # index: [unit_idx, start_idx]
    # seq 0 from unit 0
    # seq 1 from unit 0
    # seq 0 from unit 1
    index = np.array([[0, 0], [0, 1], [1, 0]])

    # blocks: (n_sequences, n_features)
    blocks = np.array(
        [
            [1.1, 1.2],  # Corresponds to index [0, 0]
            [2.1, 2.2],  # Corresponds to index [0, 1]
            [3.1, 3.2],  # Corresponds to index [1, 0]
        ]
    )

    result = t._reconstruct_ragged_array(index, blocks)

    # Unit 0 gets 2 blocks, Unit 1 gets 1 block
    expected = ak.Array([[[1.1, 1.2], [2.1, 2.2]], [[3.1, 3.2]]])  # Unit 0  # Unit 1
    # We must also regularize the expected array to match the function's output
    expected = ak_regularize_regular_axes(expected)

    # Use direct assertion, not ak.all()
    assert ak.almost_equal(result, expected)


# --- Main Integration Test ---


def test_transform_data_integration(transform_fixture):
    t = transform_fixture  # Mocked transform, batch_size=2, seq_len=4, stride=1

    # 1. Define Input Data (using awkward arrays)

    # --- THIS IS THE FIX (Part 1) ---
    # We now use ONE unit with 6 cycles.
    # This will create 2 valid sequences (stride=1):
    # - Seq 0 (start 0): needs points 0-4. Valid.
    # - Seq 1 (start 1): needs points 1-5. Valid.
    features_ak = ak.Array(
        [
            [
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
                [4.0, 4.1],
                [5.0, 5.1],
                [6.0, 6.1],
            ]  # Unit 0, length 6
        ]
    )

    # Targets: 2 target values per unit/cycle
    target_ak = ak.Array(
        [
            [
                [99, 9.9],
                [99, 9.8],
                [99, 9.7],
                [99, 9.6],
                [99, 9.5],
                [99, 9.4],
            ]  # Unit 0, length 6
        ]
    )

    # Regularize the input arrays so the sequencer sees axis=2 as regular
    features_ak = ak_regularize_regular_axes(features_ak)
    target_ak = ak_regularize_regular_axes(target_ak)

    data = {"features": features_ak, "target": target_ak}

    # 2. Define Mock Model Behavior
    # We have 2 sequences, batch_size=2. Loop will run ONCE.
    # We assume the encoder returns a 2D latent vector
    mock_encoded_batch_1 = torch.tensor([[0.5, 0.6], [1.5, 1.6]], dtype=torch.float32)

    t._model.backbone.encode.side_effect = [mock_encoded_batch_1]

    # 3. Run the transform
    result = t.transform_data(data, metadata={})

    # 4. Verify the results

    # --- Verify Features ---
    # The concatenated encoded blocks should be:
    # np.array([[0.5, 0.6], [1.5, 1.6]])
    # Reconstructed, this should be:
    expected_features = ak.Array([[[0.5, 0.6], [1.5, 1.6]]])  # Unit 0
    expected_features = ak_regularize_regular_axes(expected_features)

    assert "features" in result
    assert ak.almost_equal(result["features"], expected_features)

    # --- Verify Targets ---
    # The `y` from `sq_target.sequences_batch` will have shape (2, 2, 2)
    # Seq 0 y-window: [target[3], target[4]] -> [[99, 9.6], [99, 9.5]]
    # Seq 1 y-window: [target[4], target[5]] -> [[99, 9.5], [99, 9.4]]
    #
    # _process_target_batch takes the *last* item from this window (y_batch[:, -1, :])
    # - For Seq 0: [99, 9.5] (which is target[4])
    # - For Seq 1: [99, 9.4] (which is target[5])

    # --- THIS IS THE FIX (Part 2) ---
    # Update expected_targets to match this logic
    expected_targets = ak.Array([[[99, 9.5], [99, 9.4]]])  # Unit 0
    expected_targets = ak_regularize_regular_axes(expected_targets)

    assert "target" in result
    assert ak.almost_equal(result["target"], expected_targets)

    # --- Final Sanity Check ---
    # Check that 'encode' was called ONCE
    assert t._model.backbone.encode.call_count == 1
