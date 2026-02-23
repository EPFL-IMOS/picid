import numpy as np
import torch

from picid.model.utils.tools import (
    EarlyStopping,
    dotdict,
    StandardScaler,
    adjustment,
    cal_accuracy,
)


def test_dotdict_getattr():
    d = dotdict({"a": 1, "b": 2})
    assert d.a == 1
    assert d.b == 2
    assert d.get("c", 3) == 3


def test_dotdict_setattr():
    d = dotdict()
    d.x = 10
    assert d["x"] == 10


def test_standard_scaler_transform():
    mean, std = 2.0, 3.0
    scaler = StandardScaler(mean, std)

    data = np.array([2.0, 5.0, 8.0])
    out = scaler.transform(data)

    expected = (data - mean) / std
    np.testing.assert_array_almost_equal(out, expected)


def test_standard_scaler_inverse_transform():
    mean, std = 2.0, 3.0
    scaler = StandardScaler(mean, std)

    data = np.array([0.0, 1.0, 2.0])
    transformed = scaler.transform(data)
    recovered = scaler.inverse_transform(transformed)

    np.testing.assert_array_almost_equal(recovered, data)


def test_early_stopping_saves_on_improvement(tmp_path):
    model = torch.nn.Linear(2, 1)
    es = EarlyStopping(patience=2, verbose=False, delta=0)

    es(1.0, model, str(tmp_path))
    assert (tmp_path / "checkpoint.pth").exists()

    es(0.5, model, str(tmp_path))  # improvement
    assert es.counter == 0


def test_early_stopping_increments_counter():
    model = torch.nn.Linear(2, 1)
    es = EarlyStopping(patience=3, verbose=False, delta=0)

    es(1.0, model, "/tmp")
    es(1.1, model, "/tmp")  # worse
    assert es.counter == 1
    es(1.2, model, "/tmp")  # worse again
    assert es.counter == 2
    es(1.3, model, "/tmp")  # worse again -> early stop
    assert es.early_stop


def test_adjustment():
    gt = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0])

    gt_out, pred_out = adjustment(gt, pred.copy())

    assert np.array_equal(gt_out, gt)
    # With anomaly_state logic, pred may be modified
    assert pred_out.dtype == pred.dtype


def test_cal_accuracy():
    y_pred = np.array([1, 0, 1, 1, 0])
    y_true = np.array([1, 0, 0, 1, 0])

    acc = cal_accuracy(y_pred, y_true)
    assert acc == 0.8  # 4/5 correct
