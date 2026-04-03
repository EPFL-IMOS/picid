"""Contract: shared builders return identical payloads for identical seeds."""

from test.fixtures.builders import make_regression_batch


def test_make_regression_batch_is_deterministic():
    a = make_regression_batch(seed=7)
    b = make_regression_batch(seed=7)
    assert (a["predictions"] == b["predictions"]).all()
    assert (a["targets"] == b["targets"]).all()
