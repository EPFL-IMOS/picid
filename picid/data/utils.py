import numpy as np
import torch


def to_torch_tensor(data, device=None):
    """
    Convert input data to a ``torch.Tensor`` with a dtype chosen from the input kind.

    Parameters
    ----------
    data : Any
        List, NumPy array, or tensor to convert.
    device : torch.device or str, optional
        Target device (``"cpu"``, ``"cuda"``, etc.).

    Returns
    -------
    torch.Tensor
        Tensor on the requested device.
    """
    if torch.is_tensor(data):
        # If already a tensor, just cast
        if data.dtype.is_floating_point:
            return data.to(dtype=torch.float32, device=device)
        elif data.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return data.to(dtype=torch.long, device=device)
        else:
            raise TypeError(f"Unsupported tensor dtype: {data.dtype}")

    # Convert to NumPy array without copying if possible
    arr = np.asarray(data)

    if np.issubdtype(arr.dtype, np.floating):
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    elif np.issubdtype(arr.dtype, np.integer):
        return torch.as_tensor(arr, dtype=torch.long, device=device)
    else:
        raise TypeError(f"Unsupported input dtype: {arr.dtype}")
