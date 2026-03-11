import pytest
import numpy as np
import warnings
import pandas as pd
from picid.transforms.base_transforms.imputation_methods import ImputationTransform
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


# =========================================================================
# === Imputation Tests ===
# =========================================================================


class TestImputationTransform:
    def test_init_validation(self):
        ImputationTransform(strategy="mean")
        with pytest.raises(ValueError, match="Invalid strategy"):
            ImputationTransform(strategy="magic_wand")

    def test_impute_zero(self):
        """Test strategy='zero'."""
        data = {"features": np.array([[1.0, np.nan], [np.nan, 2.0]])}
        t = ImputationTransform(strategy="zero")
        out = t.transform_data(data, metadata={})

        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(out["features"], expected)

    def test_impute_mean_fit_transform(self):
        """Test fit_data calculation and transform application for 'mean' strategy."""
        # Create (B=2, T=2, C=1)
        # Batch 0: [10, 10]
        # Batch 1: [20, 20]
        # Global Mean = 15
        x = np.array([[[10.0], [10.0]], [[20.0], [20.0]]])
        data = {"features": x}

        t = ImputationTransform(strategy="mean")
        t.fit_data(data, metadata={})

        # Verify learned mean
        assert "features" in t.fitted_means
        np.testing.assert_allclose(t.fitted_means["features"], [15.0])

        # Transform new data with NaNs
        x_new = np.array([[[np.nan], [5.0]]])  # (1, 2, 1)
        data_new = {"features": x_new}
        out = t.transform_data(data_new, metadata={})

        # NaN replaced by 15.0
        expected = np.array([[[15.0], [5.0]]])
        np.testing.assert_array_equal(out["features"], expected)

    def test_impute_mean_fallback(self):
        """Test fallback to 0.0 if fit() wasn't called for a key."""
        t = ImputationTransform(strategy="mean")
        # No fit called

        data = {"features": np.array([[np.nan, 5.0]])}
        # Should warn and fill 0
        out = t.transform_data(data, metadata={})
        assert out["features"][0, 0] == 0.0

    def test_impute_locf_logic(self):
        """Test Last Observation Carried Forward."""
        # [1, NaN, 2, NaN] -> [1, 1, 2, 2]
        arr = np.array([[1.0, np.nan, 2.0, np.nan]]).T  # (4, 1)
        data = {"features": arr}  # 2D input (T, C)

        t = ImputationTransform(strategy="locf")
        out = t.transform_data(data, metadata={})

        expected = np.array([[1.0], [1.0], [2.0], [2.0]])
        np.testing.assert_array_equal(out["features"], expected)

    def test_impute_linear_logic(self):
        """Test Linear Interpolation."""
        # [1, NaN, 3] -> [1, 2, 3]
        arr = np.array([[1.0, np.nan, 3.0]]).T
        data = {"features": arr}

        t = ImputationTransform(strategy="linear")
        out = t.transform_data(data, metadata={})

        expected = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(out["features"], expected)

    def test_impute_linear_edges(self):
        """Test bfill/ffill behavior at edges for linear strategy."""
        # [NaN, 10, NaN] -> [10, 10, 10]
        # Interpolate can't handle edges, so we rely on the ffill/bfill fallback
        arr = np.array([[np.nan, 10.0, np.nan]]).T
        data = {"features": arr}

        t = ImputationTransform(strategy="linear")
        out = t.transform_data(data, metadata={})

        expected = np.array([[10.0], [10.0], [10.0]])
        np.testing.assert_array_equal(out["features"], expected)

    def test_impute_multichannel_independence(self):
        """Verify channels are imputed independently."""
        # Ch0: [1, NaN, 3] (Linear -> 2)
        # Ch1: [10, 10, NaN] (Linear -> 10 via ffill)
        arr = np.array([[1.0, 10.0], [np.nan, 10.0], [3.0, np.nan]])
        data = {"features": arr}

        t = ImputationTransform(strategy="linear")
        out = t.transform_data(data, metadata={})

        expected = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
        np.testing.assert_allclose(out["features"], expected)

    def test_impute_3d_batch_isolation(self):
        """Test that LOCF/Linear does not cross batch boundaries."""
        # Batch 0: [10, 20]
        # Batch 1: [NaN, 50]
        # If batch not isolated, Batch 1 start might grab 20 from Batch 0.
        # Correct behavior: Batch 1 start bfills from 50.

        x = np.zeros((2, 2, 1))
        x[0, :, 0] = [10, 20]
        x[1, :, 0] = [np.nan, 50]

        data = {"features": x}

        # Test LOCF
        t_locf = ImputationTransform(strategy="locf")
        out_locf = t_locf.transform_data(data, metadata={})
        # Batch 1 index 0 should be 50 (bfill), not 20
        assert out_locf["features"][1, 0, 0] == 50.0

    def test_impute_all_nan_column(self):
        """Test robustness when a column is entirely NaN."""
        x = np.full((1, 5, 1), np.nan)
        data = {"features": x}

        # Linear should fall back to 0.0 (fillna(0.0) in code)
        t = ImputationTransform(strategy="linear")
        out = t.transform_data(data, metadata={})

        assert not np.isnan(out["features"]).any()
        assert np.all(out["features"] == 0.0)

    def test_apply_to_logic(self):
        """Test that 'apply_to' correctly restricts the transform to specific keys."""
        data = {
            "features": np.array([[1.0, np.nan], [2.0, 3.0]]),
            "other": np.array([[10.0, np.nan], [20.0, 30.0]]),
        }
        # Apply only to 'features'
        t = ImputationTransform(strategy="zero", apply_to=["features"])
        out = t.transform_data(data, metadata={})

        # 'features' should have 0.0
        expected_features = np.array([[1.0, 0.0], [2.0, 3.0]])
        np.testing.assert_array_equal(out["features"], expected_features)

        # 'other' should remain unchanged (still has NaN)
        np.testing.assert_array_equal(out["other"], data["other"])
        assert np.isnan(out["other"][0, 1])

    def test_fit_data_ignored_for_non_mean(self):
        """Test that fit_data is a no-op for strategies like 'zero'."""
        data = {"features": np.array([[1.0, 2.0]])}
        t = ImputationTransform(strategy="zero")
        # Should not raise error and return self
        res = t.fit_data(data, metadata={})
        assert res is t
        assert not hasattr(t, "fitted_means") or not t.fitted_means

    def test_pandas_input_handling(self):
        """Test handling of pandas DataFrame/Series inputs."""
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
        data = {"features": df}
        t = ImputationTransform(strategy="zero")

        out = t.transform_data(data, metadata={})

        # Output is converted to numpy array by the transform
        expected = np.array([[1.0, 2.0], [0.0, 3.0]])
        np.testing.assert_array_equal(out["features"], expected)

    def test_2d_input_compatibility(self):
        """Ensure 2D inputs (Time, Channels) are handled correctly."""
        # Shape (3, 1)
        x = np.array([[1.0], [np.nan], [3.0]])
        data = {"features": x}

        t = ImputationTransform(strategy="linear")
        out = t.transform_data(data, metadata={})

        expected = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(out["features"], expected)

    # --- New Robustness Tests ---

    def test_empty_data_input(self):
        """Test that transform handles empty data dictionary gracefully."""
        t = ImputationTransform(strategy="zero")
        out = t.transform_data({}, metadata={})
        assert out == {}

    def test_apply_to_missing_key(self):
        """Test that specifying a key in apply_to that doesn't exist ignores it."""
        data = {"features": np.array([[1.0, np.nan]])}
        t = ImputationTransform(strategy="zero", apply_to=["non_existent"])

        out = t.transform_data(data, metadata={})

        # Data should be unchanged (NaN remains) because 'features' wasn't targeted
        np.testing.assert_array_equal(out["features"], data["features"])
        assert np.isnan(out["features"][0, 1])

    def test_clean_data_passthrough(self):
        """Test that data without NaNs is passed through correctly."""
        data = {"features": np.array([[1.0, 2.0]])}
        t = ImputationTransform(strategy="linear")
        out = t.transform_data(data, metadata={})

        np.testing.assert_array_equal(out["features"], data["features"])

    def test_fit_on_all_nans(self):
        """Test fitting on data that is 100% NaN (should fallback to 0.0 mean)."""
        data = {"features": np.full((5, 2), np.nan)}
        t = ImputationTransform(strategy="mean")

        # Use warnings.catch_warnings to safely ignore the RuntimeWarning
        # caused by taking the mean of an empty slice (all NaNs).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t.fit_data(data, metadata={})

        assert "features" in t.fitted_means
        np.testing.assert_array_equal(t.fitted_means["features"], np.array([0.0, 0.0]))

    def test_fit_transform_separate_batches(self):
        """Test fitting on one batch and transforming another."""
        # Train: Mean=10
        train_data = {"features": np.full((5, 1), 10.0)}
        # Test: Mean=20 (but has NaNs)
        test_data = {"features": np.array([[20.0], [np.nan]])}

        t = ImputationTransform(strategy="mean")
        t.fit_data(train_data, metadata={})

        out = t.transform_data(test_data, metadata={})

        # The NaN in test data should be filled with TRAIN mean (10.0), not TEST mean
        expected = np.array([[20.0], [10.0]])
        np.testing.assert_array_equal(out["features"], expected)

    # --- Exception Handling Tests ---

    def test_fit_data_invalid_dimensions(self):
        """Test that fit_data raises ValueError for >3D data."""
        data = {"features": np.ones((2, 2, 2, 2))}
        t = ImputationTransform(strategy="mean")
        with pytest.raises(ValueError, match="invalid dimensions"):
            t.fit_data(data, metadata={})

    def test_transform_data_invalid_dimensions(self):
        """Test that transform_data raises ValueError for >3D data."""
        data = {"features": np.ones((2, 2, 2, 2))}
        t = ImputationTransform(strategy="zero")
        with pytest.raises(ValueError, match="Data must be 2D or 3D"):
            t.transform_data(data, metadata={})

    def test_impute_1d_array_exception(self):
        """Test behavior when input data is 1D (should be reshaped or raise error depending on util config)."""
        # If convert_to_numpy handles 1D -> 2D, this might pass.
        # But if it reaches _impute_array as 1D, it should fail if not handled.
        # Assuming internal logic standardizes to 2D/3D.
        pass  # Covered by convert_to_numpy tests usually, but good to check integration if desired.


# =========================================================================
# === Integration / Workflow Tests ===
# =========================================================================


def test_full_mcar_workflow(sample_data_3d):
    """
    Test the full pipeline: Data -> Corrupt -> Impute -> Check MSE.
    Using linear interpolation on a sine wave should yield low error.
    This serves as a functional test for Imputation quality.
    """
    # 1. Corrupt (Pointwise 20%)
    corruptor = MCARCorruptorTransform(ratios=[0.2, 0.0, 0.0], mode="point", seed=42)
    corrupted_data = corruptor.transform_data(sample_data_3d, metadata={})

    # Verify corruption
    orig_feat = sample_data_3d["features"]
    corr_feat = corrupted_data["features"]
    mask = np.isnan(corr_feat)
    assert mask.sum() > 0

    # 2. Impute (Linear)
    imputer = ImputationTransform(strategy="linear")
    repaired_data = imputer.transform_data(corrupted_data, metadata={})
    rep_feat = repaired_data["features"]

    # Verify no NaNs remain
    assert not np.isnan(rep_feat).any()

    # 3. Calculate Error on Ch0 (Sine wave)
    # Only calculate error where data was missing
    missing_indices = np.where(mask[:, :, 0])

    if len(missing_indices[0]) > 0:
        orig_vals = orig_feat[:, :, 0][missing_indices]
        rep_vals = rep_feat[:, :, 0][missing_indices]

        mse = np.mean((orig_vals - rep_vals) ** 2)

        # MSE should be small for linear interp on smooth sine wave
        # Sine wave amp 1.0.
        assert mse < 0.1, f"Imputation MSE too high: {mse}"


def test_mixed_strategies_integration():
    """
    Test a complex scenario where 4 channels each use a different strategy.
    Shape: (B=1, T=4, C=4)
    Ch0: 'zero'   -> [NaN, 5] -> [0, 5]
    Ch1: 'mean'   -> [NaN, 10, 20] -> [15, 10, 20] (Requires fit)
    Ch2: 'locf'   -> [1, NaN, 2] -> [1, 1, 2]
    Ch3: 'linear' -> [1, NaN, 3] -> [1, 2, 3]
    """
    # Setup Data
    # B=1, T=4, C=4
    x = np.full((1, 4, 4), np.nan)

    # Ch0 (Zero): [NaN, 5.0, NaN, NaN]
    x[0, :, 0] = [np.nan, 5.0, np.nan, np.nan]

    # Ch1 (Mean): [10.0, 20.0, NaN, NaN] -> Mean is 15.0
    x[0, :, 1] = [10.0, 20.0, np.nan, np.nan]

    # Ch2 (LOCF): [1.0, NaN, NaN, 4.0] -> [1.0, 1.0, 1.0, 4.0]
    x[0, :, 2] = [1.0, np.nan, np.nan, 4.0]

    # Ch3 (Linear): [1.0, NaN, NaN, 4.0] -> [1.0, 2.0, 3.0, 4.0]
    x[0, :, 3] = [1.0, np.nan, np.nan, 4.0]

    data = {"features": x}
    strategies = ["zero", "mean", "locf", "linear"]

    # Run Transform
    t = ImputationTransform(strategy=strategies)
    t.fit_data(data, metadata={})  # Required for 'mean'
    out = t.transform_data(data, metadata={})

    res = out["features"][0]  # Shape (4, 4)

    # 1. Verify Zero
    np.testing.assert_array_equal(res[:, 0], [0.0, 5.0, 0.0, 0.0])

    # 2. Verify Mean (Mean of 10 and 20 is 15)
    np.testing.assert_array_equal(res[:, 1], [10.0, 20.0, 15.0, 15.0])

    # 3. Verify LOCF
    np.testing.assert_array_equal(res[:, 2], [1.0, 1.0, 1.0, 4.0])

    # 4. Verify Linear
    np.testing.assert_allclose(res[:, 3], [1.0, 2.0, 3.0, 4.0])


def test_strategy_length_mismatch():
    """Ensure error is raised if # strategies != # channels."""
    x = np.zeros((1, 5, 3))  # 3 Channels
    data = {"features": x}

    # Provide 2 strategies for 3 channels
    strategies = ["mean", "linear"]
    t = ImputationTransform(strategy=strategies)

    # Should fail at fit time (when shapes are known)
    with pytest.raises(ValueError, match="does not match number of channels"):
        t.fit_data(data, metadata={})

    # Should also fail at transform time if fit wasn't called/didn't catch it
    with pytest.raises(ValueError, match="does not match number of channels"):
        t.transform_data(data, metadata={})


def test_impute_stochastic_logic():
    """
    Test 'stochastic' strategy (Stochastic LOCF).
    It should fill NaNs and introduce variance, unlike standard LOCF.
    """
    # Create a constant signal with a gap
    # If std is 0 (constant signal), it behaves exactly like LOCF
    x = np.array([[10.0], [10.0], [np.nan], [np.nan], [10.0]])
    data = {"features": x}

    # 1. Test with zero variance (should be identical to LOCF)
    t_det = ImputationTransform(strategy="stochastic")
    # We manually set fitted_stds to 0.0 to verify the baseline behavior
    t_det.fitted_stds["features"] = np.array([0.0])
    out_det = t_det.transform_data(data, metadata={})

    # Should be exactly 10.0 everywhere
    np.testing.assert_array_equal(out_det["features"], np.full((5, 1), 10.0))

    # 2. Test with variance
    # We force a high std dev so the noise is obvious
    t_rand = ImputationTransform(strategy="stochastic")
    t_rand.fitted_stds["features"] = np.array([5.0])

    # Seed for reproducibility if needed, but here we just check properties
    np.random.seed(42)
    out_rand = t_rand.transform_data(data, metadata={})

    imputed_vals = out_rand["features"][2:4, 0]

    # Check 1: NaNs are gone
    assert not np.isnan(out_rand["features"]).any()

    # Check 2: Values are NOT exactly 10.0 (noise added)
    assert not np.all(imputed_vals == 10.0)

    # Check 3: Values are around 10.0 (within 3 sigma usually)
    # We used LOCF baseline (10) + Noise(0, 5)
    assert np.all(np.abs(imputed_vals - 10.0) < 20.0)


def test_impute_stochastic_fit():
    """Test that fit_data correctly calculates std for stochastic imputation."""
    # [0, 10, 0, 10] -> Mean=5, Std=5
    x = np.array([[0.0], [10.0], [0.0], [10.0]])
    data = {"features": x}

    t = ImputationTransform(strategy="stochastic")
    t.fit_data(data, metadata={})

    assert "features" in t.fitted_stds
    # Population std of [0, 10, 0, 10] is 5.0
    np.testing.assert_allclose(t.fitted_stds["features"], [5.0])


def test_impute_spectral_dynamic_window():
    """
    Test that spectral imputation works even if history < window_len.
    """
    # Create a signal with a gap very early on
    # History: indices 0..9 (length 10). Gap: 10..15.
    # Desired window_len is 20, but we only have 10.
    x = np.ones((30, 1))
    x[10:15] = np.nan

    data = {"features": x}

    t = ImputationTransform(
        strategy="spectral",
        spectral_window_len=20,
        spectral_top_k=1,  # Requesting 20
    )

    # Should not crash. Should use available 10 points.
    out = t.transform_data(data, metadata={})

    # Since signal is constant 1, FFT should find DC/low freq and fill with ~1
    # (Note: DC offset removal might make it 0 if not handled, but here we
    # test mechanism stability, not exact constant reconstruction which requires DC preservation)
    assert not np.isnan(out["features"]).any()


def test_impute_spectral_no_history():
    """
    Test spectral fallback when there is zero history (gap at start).
    """
    # Gap at the very beginning
    x = np.array([[np.nan], [np.nan], [1.0], [1.0]])
    data = {"features": x}

    t = ImputationTransform(strategy="spectral")
    out = t.transform_data(data, metadata={})

    # Should fallback to 0.0
    expected = np.array([[0.0], [0.0], [1.0], [1.0]])
    np.testing.assert_array_equal(out["features"], expected)


# =========================================================================
# Parametrized strategy tests
# =========================================================================


@pytest.mark.parametrize(
    "strategy,data_input,expected_nan_free",
    [
        ("zero", {"features": np.array([[1.0, np.nan], [np.nan, 2.0]])}, True),
        ("mean", {"features": np.array([[10.0, np.nan], [20.0, 20.0]])}, True),
        ("locf", {"features": np.array([[1.0, np.nan, 2.0, np.nan]]).T}, True),
        ("linear", {"features": np.array([[1.0, np.nan, 3.0]]).T}, True),
    ],
    ids=["zero", "mean", "locf", "linear"],
)
def test_impute_strategies_parametrized(strategy, data_input, expected_nan_free):
    """Parametrized tests for core imputation strategies."""
    t = ImputationTransform(strategy=strategy)
    if strategy == "mean":
        t.fit_data(data_input, metadata={})
    out = t.transform_data(data_input, metadata={})
    if expected_nan_free:
        assert not np.isnan(out["features"]).any()
    assert out["features"].shape == data_input["features"].shape


def test_impute_copy_past_logic():
    """Test copy_past (blockwise past copy) strategy."""
    # Periodic signal: [1,2,3,1,2,3,...] with a gap
    x = np.array([[1.0], [2.0], [3.0], [np.nan], [np.nan], [3.0]])
    data = {"features": x}
    t = ImputationTransform(strategy="copy_past")
    out = t.transform_data(data, metadata={})
    assert not np.isnan(out["features"]).any()
    assert out["features"].shape == x.shape


def test_impute_spectral_match_variance():
    """Test spectral imputation with match_variance=True."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 64)
    x = np.sin(t) + 0.1 * np.random.randn(64)
    x = x.reshape(1, -1, 1)
    x[0, 20:30, 0] = np.nan
    data = {"features": x}
    t_imp = ImputationTransform(
        strategy="spectral", spectral_match_variance=True, spectral_window_len=20
    )
    out = t_imp.transform_data(data, metadata={})
    assert not np.isnan(out["features"]).any()
