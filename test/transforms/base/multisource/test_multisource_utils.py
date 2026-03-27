"""Tests for picid.transforms.base.multisource.utils."""

import numpy as np
from unittest.mock import MagicMock

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.multisource.utils import _log_transform_error


def test_log_transform_error_runs_without_raise():
    """_log_transform_error runs without raising when given a small segment."""
    segment = NamedTransformInput(x=np.zeros((2, 3)))
    meta = {"apply_to_keys": ["x"]}
    mock_transform = MagicMock(__class__=MagicMock(__name__="FakeTransform"))
    _log_transform_error(mock_transform, [segment], meta, "dense", "dense")
