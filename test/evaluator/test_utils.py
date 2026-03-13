"""
Tests for picid.evaluator.utils module.

This module contains tests for utility metric functions used in PHM evaluators:
- RSE (Relative Squared Error)
- CORR (Correlation)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- MSPE (Mean Squared Percentage Error)
- metric (aggregate function)

These metrics are fundamental for evaluating RUL predictions and sensor
health assessments in Prognostics and Health Management (PHM) systems.

Coverage Target: 100%
"""

import torch
from numpy.testing import assert_almost_equal

from picid.evaluator.utils import (
    RSE,
    CORR,
    MAE,
    MSE,
    RMSE,
    MAPE,
    MSPE,
    metric,
)


# =============================================================================
# === PHM Mock Data Generators ===
# =============================================================================


def create_nominal_rul_data():
    """
    Create nominal RUL prediction data representing healthy degradation trends.

    **PHM Context**: In nominal conditions, RUL predictions closely follow
    actual degradation curves with minimal deviation. This simulates a
    well-calibrated prognostics model on equipment with predictable wear.

    Returns:
        tuple: (predictions, targets) as torch tensors
    """
    # Simulated RUL values from 100 cycles down to 0
    targets = torch.linspace(100.0, 0.0, steps=50)
    # Predictions with small noise (well-calibrated model)
    predictions = targets + torch.randn_like(targets) * 2.0
    return predictions, targets


def create_fault_signature_data():
    """
    Create fault signature data simulating degradation onset detection.

    **PHM Context**: Fault signatures show sudden deviations from nominal
    behavior. A good prognostics model should detect and track these
    changes in the RUL predictions.

    Returns:
        tuple: (predictions, targets) as torch tensors
    """
    # Target shows sudden degradation after cycle 25
    targets = torch.cat(
        [
            torch.full((25,), 80.0),  # Healthy phase
            torch.linspace(80.0, 0.0, steps=25),  # Rapid degradation
        ]
    )
    # Predictions lag behind (typical model behavior at fault onset)
    predictions = torch.cat(
        [
            torch.full((30,), 85.0),  # Model still sees healthy
            torch.linspace(70.0, 5.0, steps=20),  # Catches up
        ]
    )
    return predictions, targets


def create_perfect_prediction_data():
    """
    Create perfect prediction data for baseline testing.

    **PHM Context**: Perfect predictions establish the theoretical
    optimum for all metrics (zero error metrics, perfect correlation).

    Returns:
        tuple: (predictions, targets) as torch tensors
    """
    targets = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    predictions = targets.clone()
    return predictions, targets


def create_multivariate_sensor_data():
    """
    Create multivariate sensor data for correlation testing.

    **PHM Context**: Multivariate data represents multiple sensors
    measuring different aspects of equipment health (e.g., temperature,
    vibration, pressure). Correlation analysis helps identify
    sensor relationships and redundancy.

    Returns:
        tuple: (predictions, targets) as 2D torch tensors (samples, features)
    """
    # 10 samples, 3 sensors
    targets = torch.randn(10, 3)
    # Predictions with varying correlation per sensor
    predictions = targets + torch.randn(10, 3) * 0.1
    return predictions, targets


# =============================================================================
# === Test Classes for Metric Functions ===
# =============================================================================


class TestRSE:
    """Tests for Relative Squared Error (RSE) metric.

    RSE measures the ratio of prediction error to target variance,
    useful for comparing model performance across different scales.
    """

    def test_rse_perfect_prediction(self):
        """Test RSE returns 0 for perfect predictions.

        **PHM Logic**: Perfect RUL predictions yield RSE=0.

        **Methodology**: Use identical predictions and targets.

        **Expected**: RSE = 0.0

        Validates: Requirement EV-RSE-1 - RSE lower bound
        """
        pred, true = create_perfect_prediction_data()
        result = RSE(pred, true)

        assert_almost_equal(result.item(), 0.0, decimal=5)

    def test_rse_nominal_data(self):
        """Test RSE with nominal RUL predictions.

        **PHM Logic**: Well-calibrated models have low RSE (<1.0).

        **Methodology**: Generate nominal degradation data with small noise.

        **Expected**: RSE is a small positive value.

        Validates: Requirement EV-RSE-2 - RSE for good predictions
        """
        pred, true = create_nominal_rul_data()
        result = RSE(pred, true)

        assert result.item() > 0.0, "RSE should be positive for imperfect predictions"
        assert (
            result.item() < 1.0
        ), "RSE < 1 indicates model is better than mean prediction"

    def test_rse_fault_signature(self):
        """Test RSE detects degraded predictions at fault onset.

        **PHM Logic**: At fault onset, prediction lag causes higher RSE.

        **Methodology**: Use fault signature data with prediction delay.

        **Expected**: RSE is higher than nominal case.

        Validates: Requirement EV-RSE-3 - RSE sensitivity to faults
        """
        pred, true = create_fault_signature_data()
        result = RSE(pred, true)

        assert result.item() > 0.1, "RSE should be elevated for fault signature data"


class TestCORR:
    """Tests for Correlation (CORR) metric.

    CORR measures linear correlation between predictions and targets,
    critical for validating sensor data consistency and model calibration.
    """

    def test_corr_perfect_correlation(self):
        """Test CORR returns high value for perfectly correlated data.

        **PHM Logic**: Perfect correlation indicates ideal model calibration.

        **Methodology**: Use scaled predictions (perfectly correlated).

        **Expected**: CORR close to 1.0

        Validates: Requirement EV-CORR-1 - CORR upper bound
        """
        # Create data with perfect linear relationship but different scales
        true = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = true * 2.0  # Perfect correlation but different scale
        result = CORR(pred, true)

        # CORR should be very high for perfect linear relationship
        assert result.item() > 0.99, "CORR should be ~1.0 for perfectly correlated data"

    def test_corr_high_correlation(self):
        """Test CORR with highly correlated predictions.

        **PHM Logic**: Good PHM models show CORR > 0.9.

        **Methodology**: Add small noise to perfect predictions.

        **Expected**: CORR close to 1.0.

        Validates: Requirement EV-CORR-2 - CORR for good predictions
        """
        true = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = true + torch.randn_like(true) * 0.5
        result = CORR(pred, true)

        assert (
            result.item() > 0.9
        ), "Correlation should be high for well-calibrated model"

    def test_corr_multivariate(self):
        """Test CORR with multivariate sensor data.

        **PHM Logic**: Multivariate correlation tracks sensor relationships.

        **Methodology**: Use multivariate data with per-feature correlation.

        **Expected**: Mean correlation across features.

        Validates: Requirement EV-CORR-3 - Multivariate correlation
        """
        pred, true = create_multivariate_sensor_data()
        result = CORR(pred, true)

        # Result should be mean correlation
        assert result.ndim == 0 or result.shape[0] == 1, "CORR should return scalar"


class TestMAE:
    """Tests for Mean Absolute Error (MAE) metric.

    MAE provides intuitive error magnitude in original units,
    critical for RUL predictions in PHM applications.
    """

    def test_mae_perfect_prediction(self):
        """Test MAE returns 0 for perfect predictions.

        **PHM Logic**: Perfect RUL predictions yield MAE=0.

        **Methodology**: Use identical predictions and targets.

        **Expected**: MAE = 0.0

        Validates: Requirement EV-MAE-1 - MAE lower bound
        """
        pred, true = create_perfect_prediction_data()
        result = MAE(pred, true)

        assert_almost_equal(result.item(), 0.0, decimal=5)

    def test_mae_known_error(self):
        """Test MAE with known error values.

        **PHM Logic**: MAE measures average absolute RUL prediction error.

        **Methodology**: Create data with known errors for verification.

        **Expected**: MAE = 2.0 (average of |3-1|, |6-4|, |9-7|)

        Validates: Requirement EV-MAE-2 - MAE calculation correctness
        """
        pred = torch.tensor([3.0, 6.0, 9.0])
        true = torch.tensor([1.0, 4.0, 7.0])

        result = MAE(pred, true)

        # |3-1| + |6-4| + |9-7| = 2+2+2 = 6, mean = 2.0
        assert_almost_equal(result.item(), 2.0, decimal=5)

    def test_mae_asymmetric_errors(self):
        """Test MAE handles positive and negative errors equally.

        **PHM Logic**: Overestimation and underestimation of RUL are
        equally penalized by MAE (unlike NASA scoring).

        **Methodology**: Mix positive and negative prediction errors.

        **Expected**: MAE = mean(|errors|)

        Validates: Requirement EV-MAE-3 - MAE symmetry
        """
        pred = torch.tensor([5.0, 15.0, 25.0])  # Mixed over/under
        true = torch.tensor([10.0, 10.0, 10.0])  # Constant target

        result = MAE(pred, true)

        # |-5| + |5| + |15| = 5+5+15 = 25, mean = 8.33...
        expected = (5.0 + 5.0 + 15.0) / 3.0
        assert_almost_equal(result.item(), expected, decimal=5)


class TestMSE:
    """Tests for Mean Squared Error (MSE) metric.

    MSE penalizes large errors more than small errors,
    important for detecting outliers in RUL predictions.
    """

    def test_mse_perfect_prediction(self):
        """Test MSE returns 0 for perfect predictions.

        **PHM Logic**: Perfect RUL predictions yield MSE=0.

        **Methodology**: Use identical predictions and targets.

        **Expected**: MSE = 0.0

        Validates: Requirement EV-MSE-1 - MSE lower bound
        """
        pred, true = create_perfect_prediction_data()
        result = MSE(pred, true)

        assert_almost_equal(result.item(), 0.0, decimal=5)

    def test_mse_known_error(self):
        """Test MSE with known error values.

        **PHM Logic**: MSE squares errors before averaging.

        **Methodology**: Create data with known errors for verification.

        **Expected**: MSE = mean((3-1)^2, (6-4)^2, (9-7)^2) = 4.0

        Validates: Requirement EV-MSE-2 - MSE calculation correctness
        """
        pred = torch.tensor([3.0, 6.0, 9.0])
        true = torch.tensor([1.0, 4.0, 7.0])

        result = MSE(pred, true)

        # (3-1)^2 + (6-4)^2 + (9-7)^2 = 4+4+4 = 12, mean = 4.0
        assert_almost_equal(result.item(), 4.0, decimal=5)

    def test_mse_penalizes_outliers(self):
        """Test MSE heavily penalizes large errors.

        **PHM Logic**: Large RUL prediction errors (e.g., missing
        imminent failures) are severely penalized by MSE.

        **Methodology**: Compare MSE with/without outlier.

        **Expected**: Single outlier significantly increases MSE.

        Validates: Requirement EV-MSE-3 - MSE outlier sensitivity
        """
        true = torch.tensor([10.0, 10.0, 10.0])

        # No outlier
        pred_normal = torch.tensor([11.0, 11.0, 11.0])
        mse_normal = MSE(pred_normal, true)

        # With outlier (large error on third sample)
        pred_outlier = torch.tensor([11.0, 11.0, 50.0])
        mse_outlier = MSE(pred_outlier, true)

        assert (
            mse_outlier.item() > mse_normal.item() * 10
        ), "MSE should heavily penalize outliers"


class TestRMSE:
    """Tests for Root Mean Squared Error (RMSE) metric.

    RMSE provides error in original units while maintaining
    MSE's outlier sensitivity.
    """

    def test_rmse_perfect_prediction(self):
        """Test RMSE returns 0 for perfect predictions.

        **PHM Logic**: Perfect RUL predictions yield RMSE=0.

        **Methodology**: Use identical predictions and targets.

        **Expected**: RMSE = 0.0

        Validates: Requirement EV-RMSE-1 - RMSE lower bound
        """
        pred, true = create_perfect_prediction_data()
        result = RMSE(pred, true)

        assert_almost_equal(result.item(), 0.0, decimal=5)

    def test_rmse_is_sqrt_mse(self):
        """Test RMSE equals square root of MSE.

        **PHM Logic**: RMSE = sqrt(MSE) by definition.

        **Methodology**: Calculate both and compare.

        **Expected**: RMSE == sqrt(MSE)

        Validates: Requirement EV-RMSE-2 - RMSE definition
        """
        pred, true = create_nominal_rul_data()

        mse_result = MSE(pred, true)
        rmse_result = RMSE(pred, true)

        expected_rmse = torch.sqrt(mse_result)
        assert_almost_equal(rmse_result.item(), expected_rmse.item(), decimal=5)


class TestMAPE:
    """Tests for Mean Absolute Percentage Error (MAPE) metric.

    MAPE provides scale-independent error measure, useful for
    comparing across different equipment types in PHM.
    """

    def test_mape_perfect_prediction(self):
        """Test MAPE returns 0 for perfect predictions.

        **PHM Logic**: Perfect RUL predictions yield MAPE=0%.

        **Methodology**: Use identical predictions and targets.

        **Expected**: MAPE = 0.0

        Validates: Requirement EV-MAPE-1 - MAPE lower bound
        """
        pred, true = create_perfect_prediction_data()
        result = MAPE(pred, true)

        assert_almost_equal(result.item(), 0.0, decimal=5)

    def test_mape_known_percentage(self):
        """Test MAPE with known percentage errors.

        **PHM Logic**: MAPE measures relative error percentage.

        **Methodology**: Create data with 10% uniform error.

        **Expected**: MAPE = 0.1 (10%)

        Validates: Requirement EV-MAPE-2 - MAPE calculation correctness
        """
        true = torch.tensor([100.0, 200.0, 300.0])
        pred = true * 1.10  # 10% overestimation

        result = MAPE(pred, true)

        assert_almost_equal(result.item(), 0.1, decimal=5)


class TestMSPE:
    """Tests for Mean Squared Percentage Error (MSPE) metric.

    MSPE combines scale independence with outlier sensitivity.
    """

    def test_mspe_perfect_prediction(self):
        """Test MSPE returns 0 for perfect predictions.

        **PHM Logic**: Perfect RUL predictions yield MSPE=0.

        **Methodology**: Use identical predictions and targets.

        **Expected**: MSPE = 0.0

        Validates: Requirement EV-MSPE-1 - MSPE lower bound
        """
        pred, true = create_perfect_prediction_data()
        result = MSPE(pred, true)

        assert_almost_equal(result.item(), 0.0, decimal=5)

    def test_mspe_known_percentage(self):
        """Test MSPE with known percentage errors.

        **PHM Logic**: MSPE squares percentage errors.

        **Methodology**: Create data with 10% uniform error.

        **Expected**: MSPE = 0.01 (10%^2)

        Validates: Requirement EV-MSPE-2 - MSPE calculation correctness
        """
        true = torch.tensor([100.0, 200.0, 300.0])
        pred = true * 1.10  # 10% overestimation

        result = MSPE(pred, true)

        # (0.1)^2 = 0.01
        assert_almost_equal(result.item(), 0.01, decimal=5)


class TestMetricFunction:
    """Tests for the aggregate metric() function.

    This function computes all metrics at once for efficiency.
    """

    def test_metric_returns_all_values(self):
        """Test metric() returns tuple of all 5 metrics.

        **PHM Logic**: Aggregate function for efficient batch evaluation.

        **Methodology**: Call metric() and verify tuple structure.

        **Expected**: Tuple of (MAE, MSE, RMSE, MAPE, MSPE)

        Validates: Requirement EV-MET-1 - Aggregate metric function
        """
        pred, true = create_nominal_rul_data()

        result = metric(pred, true)

        assert isinstance(result, tuple), "metric() should return tuple"
        assert len(result) == 5, "metric() should return 5 values"

        mae, mse, rmse, mape, mspe = result

        # Verify each is a scalar tensor
        for val in result:
            assert hasattr(val, "item"), "Each metric should be a tensor"

    def test_metric_perfect_prediction(self):
        """Test metric() with perfect predictions.

        **PHM Logic**: All metrics should be zero for perfect predictions.

        **Methodology**: Use identical predictions and targets.

        **Expected**: All values = 0.0

        Validates: Requirement EV-MET-2 - Aggregate metric lower bound
        """
        pred, true = create_perfect_prediction_data()

        mae, mse, rmse, mape, mspe = metric(pred, true)

        assert_almost_equal(mae.item(), 0.0, decimal=5)
        assert_almost_equal(mse.item(), 0.0, decimal=5)
        assert_almost_equal(rmse.item(), 0.0, decimal=5)
        assert_almost_equal(mape.item(), 0.0, decimal=5)
        assert_almost_equal(mspe.item(), 0.0, decimal=5)

    def test_metric_consistency(self):
        """Test metric() returns consistent values with individual functions.

        **PHM Logic**: Aggregate function must match individual calculations.

        **Methodology**: Compare aggregate vs individual function calls.

        **Expected**: Values match within numerical precision.

        Validates: Requirement EV-MET-3 - Aggregate metric consistency
        """
        pred, true = create_fault_signature_data()

        mae_agg, mse_agg, rmse_agg, mape_agg, mspe_agg = metric(pred, true)

        mae_ind = MAE(pred, true)
        mse_ind = MSE(pred, true)
        rmse_ind = RMSE(pred, true)
        mape_ind = MAPE(pred, true)
        mspe_ind = MSPE(pred, true)

        assert_almost_equal(mae_agg.item(), mae_ind.item(), decimal=5)
        assert_almost_equal(mse_agg.item(), mse_ind.item(), decimal=5)
        assert_almost_equal(rmse_agg.item(), rmse_ind.item(), decimal=5)
        assert_almost_equal(mape_agg.item(), mape_ind.item(), decimal=5)
        assert_almost_equal(mspe_agg.item(), mspe_ind.item(), decimal=5)
