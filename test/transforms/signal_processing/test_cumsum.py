import numpy as np
import awkward as ak
import pytest

from picid.data.data_objects import NamedTransformInput
from picid.transforms.signal_processing.cumsum import CumSumTransform


def test_cumsum_transform_apply_regular_3d():
    """CumSumTransform applies cumsum of squared values along the last axis (features)."""
    transform = CumSumTransform()

    # Regular 3D: (units=2, time=4, features=3)
    x = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]],
            [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        ]
    )
    data = NamedTransformInput(features=ak.Array(x))
    metadata = {}

    result = transform.transform_data(data, metadata)

    # Transform: flatten merges axis 0,1; cumsum(axis=1) along features
    expected = np.cumsum(x.reshape(-1, 3) ** 2, axis=1).reshape(2, 4, 3)
    result_arr = np.asarray(result)
    np.testing.assert_allclose(result_arr, expected)


def test_cumsum_transform_single_unit():
    """CumSumTransform with single unit - cumsum along last (feature) axis."""
    transform = CumSumTransform()

    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
    data = NamedTransformInput(features=ak.Array(x))
    metadata = {}

    result = transform.transform_data(data, metadata)

    # Per row (time step): cumsum of squares along features
    expected = np.cumsum(x[0] ** 2, axis=1)  # [[1,5],[9,25],[25,61]]
    result_arr = np.asarray(result)
    np.testing.assert_allclose(result_arr, expected.reshape(1, 3, 2))


def test_cumsum_transform_ragged_input():
    """CumSumTransform with ragged (variable-length) sequences."""
    transform = CumSumTransform()

    # 2 units: first has 3 time steps, second has 2
    ragged = ak.Array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0]],
        ]
    )
    data = NamedTransformInput(features=ragged)
    metadata = {}

    result = transform.transform_data(data, metadata)

    assert len(result) == 2
    assert len(result[0]) == 3
    assert len(result[1]) == 2

    # Per row: cumsum of squares along the feature axis
    u0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected0 = np.cumsum(u0**2, axis=1)  # [[1,5],[9,25],[25,61]]
    np.testing.assert_allclose(np.asarray(result[0]), expected0)


def test_cumsum_transform_requires_single_key():
    with pytest.raises(AssertionError, match="exactly one entry"):
        transform = CumSumTransform()
        data = NamedTransformInput(
            features=np.array([[[1.0]]]),
            target=np.array([[[1.0]]]),
        )
        transform.transform_data(data, {})


def test_cumsum_transform_requires_3d():
    with pytest.raises(AssertionError, match="3-dimensional"):
        transform = CumSumTransform()
        data = NamedTransformInput(features=np.array([[1.0, 2.0], [3.0, 4.0]]))
        transform.transform_data(data, {})
