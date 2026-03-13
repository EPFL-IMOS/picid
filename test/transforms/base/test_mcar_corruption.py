import pytest
import numpy as np
from picid.transforms.base_transforms.mcar_corruption import MCARCorruptorTransform

# --- Fixtures ---


@pytest.fixture
def sample_data_3d():
    """
    Creates a standard 3D dataset (Batch, Time, Channels).
    Shape: (5, 100, 3)
    """
    B, T, C = 5, 100, 3
    # Create deterministic signals:
    # Ch0: Sine wave
    # Ch1: Constant 10
    # Ch2: Linear Ramp
    t = np.linspace(0, 4 * np.pi, T)

    ch0 = np.sin(t)
    ch1 = np.full(T, 10.0)
    ch2 = np.linspace(0, 10, T)

    # Broadcast to batch
    batch_data = np.zeros((B, T, C))
    for b in range(B):
        batch_data[b, :, 0] = ch0
        batch_data[b, :, 1] = ch1
        batch_data[b, :, 2] = ch2

    return {
        "features": batch_data,
        "target": np.random.rand(B, T, 1),  # Should remain untouched
    }


@pytest.fixture
def sample_data_2d():
    """
    Creates a 2D dataset (Time, Channels).
    Shape: (50, 2)
    """
    T, C = 50, 2
    data = np.random.randn(T, C)
    return {"features": data}


# =========================================================================
# === MCAR Corruptor Tests ===
# =========================================================================


class TestMCARCorruptor:
    def test_init_validation(self):
        """Test strict initialization validation."""
        # Valid
        MCARCorruptorTransform(ratios=[0.1], mode="point")
        MCARCorruptorTransform(
            ratios=[0.1], mode="block", block_params={"min_size": 2, "max_size": 5}
        )
        # Valid Proportional
        MCARCorruptorTransform(
            ratios=[0.1], mode="block", block_params={"min_size": 0.05, "max_size": 0.1}
        )

        # Invalid Mode
        with pytest.raises(ValueError, match="Invalid mode"):
            MCARCorruptorTransform(ratios=[0.1], mode="invalid")

        # Invalid Block Params (Missing keys)
        with pytest.raises(ValueError, match="must contain 'min_size'"):
            MCARCorruptorTransform(
                ratios=[0.1], mode="block", block_params={"min_size": 5}
            )

        # Invalid Block Params (min > max for int)
        with pytest.raises(ValueError, match="cannot be greater"):
            MCARCorruptorTransform(
                ratios=[0.1], mode="block", block_params={"min_size": 10, "max_size": 5}
            )

        # Invalid Block Params (min > max for float)
        with pytest.raises(ValueError, match="cannot be greater"):
            MCARCorruptorTransform(
                ratios=[0.1],
                mode="block",
                block_params={"min_size": 0.5, "max_size": 0.1},
            )

    def test_corrupt_ratio_accuracy(self, sample_data_3d):
        """Test if the point-wise corruption roughly respects the requested ratio."""
        # Request 50% missing on Channel 0
        t = MCARCorruptorTransform(
            ratios=[0.5, 0.0, 0.0], mode="point", seed=42, apply_to=["features"]
        )
        out = t.transform_data(sample_data_3d, metadata={})

        features = out["features"]

        # Check Ch0 (Target 50%)
        # Exact count check is deterministic due to seed
        n_missing_ch0 = np.isnan(features[:, :, 0]).sum()
        total_ch0 = features.shape[0] * features.shape[1]
        assert n_missing_ch0 == int(total_ch0 * 0.5)

        # Check Ch1 (Target 0%)
        assert np.isnan(features[:, :, 1]).sum() == 0

        # Check Ch2 (Default 0% since list was len 1?) No, list was len 3.
        assert np.isnan(features[:, :, 2]).sum() == 0

    def test_corrupt_block_constraints(self, sample_data_3d):
        """
        Test that in 'block' mode, no gap is smaller than min_size.
        """
        min_s, max_s = 5, 10
        t = MCARCorruptorTransform(
            ratios=[0.3, 0.0, 0.0],
            mode="block",
            block_params={"min_size": min_s, "max_size": max_s},
            seed=123,
            apply_to=["features"],
        )
        out = t.transform_data(sample_data_3d, metadata={})

        # Analyze Batch 0, Channel 0
        signal = out["features"][0, :, 0]
        mask = np.isnan(signal).astype(int)

        # Run Length Encoding on the mask
        # Pad with 0 to detect edges
        bounded = np.concatenate(([0], mask, [0]))
        diffs = np.diff(bounded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts

        if len(run_lengths) > 0:
            assert np.all(
                run_lengths >= min_s
            ), f"Found blocks smaller than {min_s}: {run_lengths}"

    def test_corrupt_block_constraints_proportional(self, sample_data_3d):
        """
        Test that in 'block' mode with proportional sizes, gaps respect T * proportion.
        """
        # T = 100. Min prop 0.05 -> 5 steps. Max prop 0.1 -> 10 steps.
        min_p, max_p = 0.05, 0.1
        t = MCARCorruptorTransform(
            ratios=[0.3, 0.0, 0.0],
            mode="block",
            block_params={"min_size": min_p, "max_size": max_p},
            seed=123,
            apply_to=["features"],
        )
        out = t.transform_data(sample_data_3d, metadata={})

        # Expected min size in absolute steps
        T = sample_data_3d["features"].shape[1]  # 100
        expected_min_s = int(T * min_p)

        # Analyze Batch 0, Channel 0
        signal = out["features"][0, :, 0]
        mask = np.isnan(signal).astype(int)

        # Run Length Encoding on the mask
        bounded = np.concatenate(([0], mask, [0]))
        diffs = np.diff(bounded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts

        if len(run_lengths) > 0:
            assert np.all(
                run_lengths >= expected_min_s
            ), f"Found blocks smaller than {expected_min_s}: {run_lengths}"

    def test_corrupt_integrity(self, sample_data_3d):
        """Ensure keys not in 'apply_to' are untouched."""
        t = MCARCorruptorTransform(ratios=[0.5], apply_to=["features"])

        original_target = sample_data_3d["target"].copy()
        out = t.transform_data(sample_data_3d, metadata={})

        # Features should change
        assert np.isnan(out["features"]).any()

        # Target should be identical object content
        np.testing.assert_array_equal(out["target"], original_target)

    def test_corrupt_2d_input(self, sample_data_2d):
        """Test handling of (T, C) inputs."""
        t = MCARCorruptorTransform(ratios=[0.5, 1.0], mode="point")
        out = t.transform_data(sample_data_2d, metadata={})

        data = out["features"]
        assert data.ndim == 2
        # Ch0: 50%
        assert np.isnan(data[:, 0]).any()
        # Ch1: 100%
        assert np.all(np.isnan(data[:, 1]))

    def test_corrupt_edge_cases(self, sample_data_3d):
        """Test Ratio 0.0 and 1.0."""
        t = MCARCorruptorTransform(ratios=[0.0, 1.0, -0.5])  # -0.5 should be ignored/0
        out = t.transform_data(sample_data_3d, metadata={})
        f = out["features"]

        # Ch0: 0.0 -> No NaNs
        assert not np.isnan(f[:, :, 0]).any()
        # Ch1: 1.0 -> All NaNs
        assert np.all(np.isnan(f[:, :, 1]))
        # Ch2: <0 -> No NaNs
        assert not np.isnan(f[:, :, 2]).any()

    def test_corrupt_ratio_invalid_runtime(self, sample_data_3d):
        """Test that ratio > 1.0 raises ValueError during transform."""
        t = MCARCorruptorTransform(ratios=[1.5], mode="point")
        with pytest.raises(ValueError, match="Ratio must be <= 1.0"):
            t.transform_data(sample_data_3d, metadata={})

    def test_corrupt_short_timeseries_clamping(self):
        """Test block mode when T < min_size (should clamp block size to T)."""
        # T=3, but min_size=5
        data = {"features": np.random.randn(2, 3, 1)}
        t = MCARCorruptorTransform(
            ratios=[0.5],
            mode="block",
            block_params={"min_size": 5, "max_size": 10},
        )
        # Should run without error and corrupt something
        out = t.transform_data(data, metadata={})
        # With T=3 and min=5-> clamped to 3. Start index always 0.
        # Should likely drop everything or close to it depending on loop count
        assert np.isnan(out["features"]).any()

    def test_corrupt_high_density_overlap_retries(self, sample_data_3d):
        """Test high missingness to force overlap retries in block mode."""
        # 90% missingness will force the loop to retry finding empty spots
        t = MCARCorruptorTransform(
            ratios=[0.9],
            mode="block",
            block_params={"min_size": 5, "max_size": 10},
            seed=42,
        )
        out = t.transform_data(sample_data_3d, metadata={})
        n_nan = np.isnan(out["features"][:, :, 0]).sum()
        total = out["features"].shape[0] * out["features"].shape[1]

        # Verify we achieved high corruption despite overlaps
        assert n_nan / total > 0.85

    def test_apply_to_missing_key(self, sample_data_3d):
        """Test applying to a key that doesn't exist."""
        t = MCARCorruptorTransform(ratios=[0.5], apply_to=["ghost_key"])
        out = t.transform_data(sample_data_3d, metadata={})

        # 'features' should remain untouched (no NaNs)
        assert not np.isnan(out["features"]).any()

    def test_empty_data(self):
        """Test empty dictionary input."""
        t = MCARCorruptorTransform(ratios=[0.5])
        out = t.transform_data({}, metadata={})
        assert out == {}

    def test_invalid_input_dimensions(self):
        """Test that 4D input raises ValueError."""
        # This covers the explicit ValueError raised in _corrupt_array for ndim > 3
        data = {"features": np.random.randn(2, 10, 3, 1)}  # 4D input
        t = MCARCorruptorTransform(ratios=[0.5])

        with pytest.raises(ValueError, match="Data must be 2D .* or 3D"):
            t.transform_data(data, metadata={})
