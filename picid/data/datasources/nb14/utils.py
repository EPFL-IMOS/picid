import numpy as np
import awkward as ak


def flatten_cycles(x, y, valid_lengths=None):
    """
    Transform (B, L, C) into (sum(valid_lengths), C) by flattening cycles.
    Keeps track of cycle_indices.

    Parameters
    ----------
    x
        np.ndarray, shape (B, L, C)
    y
        np.ndarray, shape (B, 1)  # per-cycle labels
    valid_lengths
        list/array of ints, number of valid timesteps per cycle

    Returns
    -------
    flattened_x
        (sum(valid_lengths), C)
    flattened_y
        (sum(valid_lengths), 1)
    cycle_indices
        list of (start, end) indices per cycle
    """
    B, L, C = x.shape
    flattened_x, flattened_y, cycle_indices = [], [], []
    start = 0

    for i in range(B):
        vlen = valid_lengths[i] if valid_lengths is not None else L
        fx = x[i, :vlen, :]
        fy = np.repeat(y[i], vlen, axis=0)
        flattened_x.append(fx)
        flattened_y.append(fy)
        end = start + vlen
        cycle_indices.append((start, end))
        start = end

    return np.concatenate(flattened_x), np.concatenate(flattened_y), cycle_indices


def table_to_ak_array(x, valid_lengths=None):
    """ """
    B, L, C = x.shape
    flattened_x = []
    start = 0

    for i in range(B):
        vlen = valid_lengths[i] if valid_lengths is not None else L
        fx = x[i, :vlen, :]
        flattened_x.append(fx)
        end = start + vlen
        start = end

    return ak.to_regular(ak.Array(flattened_x), axis=-1)


# def unflatten_cycles(flattened_x, cycle_indices, win_len, flattened_y=None, pad_value=0):
#     """
#     Rebuild cycles from flattened data.
#     Optionally split into windows of length win_len (with padding if needed).

#     Args:
#         flattened_x: (total_len, C)
#         flattened_y: (total_len, 1)
#         cycle_indices: list of (start, end) indices per cycle.
#         win_len: int, split each cycle into windows
#         pad_value: value used to pad windows (default=0)

#     Returns:
#         x: np.ndarray of shape (B, n_windows, win_len, C)
#         y: np.ndarray of shape (B, 1)  # cycle labels
#     """
#     cycles_x, cycles_y = [], []
#     for (start, end) in cycle_indices:
#         cx = flattened_x[start:end]

#         if flattened_y is not None:
#             cys = flattened_y[start:end]
#             # Ensure all target values within a cycle are identical and take the first one.
#             cy = cys[0]
#             assert np.all(cys == cy), "All target values within a cycle must be the same."

#         n_windows = int(np.ceil(len(cx) / win_len))
#         padded_len = n_windows * win_len
#         # pad at end if not multiple of win_len
#         pad_amount = padded_len - len(cx)
#         if pad_amount > 0:
#             pad = np.full((pad_amount, cx.shape[1]), pad_value, dtype=cx.dtype)
#             cx = np.vstack([cx, pad])

#         # reshape into (n_windows, win_len, C)
#         cx = cx.reshape(n_windows, win_len, cx.shape[1])
#         cycles_x.append(cx)

#         if flattened_y is not None:
#             cycles_y.append(cy)  # Append the single target value for the cycle

#     # pad sequences so all have the same n_windows
#     max_windows = max(c.shape[0] for c in cycles_x)
#     C = flattened_x.shape[1]
#     padded_cycles = []
#     for cx in cycles_x:
#         if cx.shape[0] < max_windows:
#             pad_shape = (max_windows - cx.shape[0], win_len, C)
#             pad = np.full(pad_shape, pad_value, dtype=cx.dtype)
#             cx = np.vstack([cx, pad])
#         padded_cycles.append(cx)

#     x = np.stack(padded_cycles)  # (B, n_windows, win_len, C)

#     if flattened_y is not None:
#         y = np.array(cycles_y)  # (B, target_dims)
#     else:
#         y = None

#     return x, y
