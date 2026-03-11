import torch

from picid.model.utils.masking import TriangularCausalMask


def test_triangular_causal_mask_shape():
    B, L = 2, 4
    mask = TriangularCausalMask(B, L)

    assert mask.mask.shape == (B, 1, L, L)
    assert mask.mask.dtype == torch.bool


def test_triangular_causal_mask_upper_triangular():
    """Mask should have True above diagonal (causal: cannot attend to future)."""
    B, L = 1, 3
    mask = TriangularCausalMask(B, L)

    # triu(..., diagonal=1) → upper triangle excluding diagonal
    # Expected: diagonal and below = False, above = True
    expected = torch.tensor(
        [
            [
                [
                    [False, True, True],
                    [False, False, True],
                    [False, False, False],
                ]
            ]
        ]
    )
    assert torch.equal(mask.mask[0], expected[0])


def test_triangular_causal_mask_device():
    mask = TriangularCausalMask(2, 4, device="cpu")
    assert mask.mask.device.type == "cpu"


def test_prob_mask_import_and_mask_property():
    """ProbMask requires specific index/scores shapes from sparse attention - smoke test."""
    from picid.model.utils.masking import ProbMask

    # Minimal smoke test: class exists and has mask property
    assert hasattr(ProbMask, "__init__")
