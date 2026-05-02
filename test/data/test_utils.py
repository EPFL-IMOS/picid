"""Tests for picid.data.utils."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from picid.data.utils import to_torch_tensor


@pytest.mark.unit
class TestToTorchTensor:
    """Tests for to_torch_tensor dtype routing."""

    def test_float_tensor_returned_as_float32(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        out = to_torch_tensor(x)
        assert out.dtype == torch.float32

    def test_int32_tensor_returned_as_long(self):
        x = torch.tensor([1, 2], dtype=torch.int32)
        out = to_torch_tensor(x)
        assert out.dtype == torch.long

    def test_int64_tensor_returned_as_long(self):
        x = torch.tensor([1, 2], dtype=torch.int64)
        out = to_torch_tensor(x)
        assert out.dtype == torch.long

    def test_int8_tensor_returned_as_long(self):
        x = torch.tensor([1, 2], dtype=torch.int8)
        out = to_torch_tensor(x)
        assert out.dtype == torch.long

    def test_unsupported_tensor_dtype_raises(self):
        x = torch.tensor([True, False], dtype=torch.bool)
        with pytest.raises(TypeError, match="Unsupported tensor dtype"):
            to_torch_tensor(x)

    def test_numpy_float_array_returned_as_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = to_torch_tensor(arr)
        assert out.dtype == torch.float32

    def test_numpy_int_array_returned_as_long(self):
        arr = np.array([1, 2], dtype=np.int32)
        out = to_torch_tensor(arr)
        assert out.dtype == torch.long

    def test_python_float_list_returned_as_float32(self):
        out = to_torch_tensor([1.0, 2.0, 3.0])
        assert out.dtype == torch.float32

    def test_python_int_list_returned_as_long(self):
        out = to_torch_tensor([1, 2, 3])
        assert out.dtype == torch.long

    def test_unsupported_numpy_dtype_raises(self):
        arr = np.array(["a", "b"])
        with pytest.raises(TypeError, match="Unsupported input dtype"):
            to_torch_tensor(arr)

    def test_device_parameter_respected(self):
        out = to_torch_tensor([1.0, 2.0], device="cpu")
        assert out.device.type == "cpu"
