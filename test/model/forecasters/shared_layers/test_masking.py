"""Tests for masking utilities used by shared baseline layers."""

import torch

from picid.model.forecasters.shared_layers.masking import (
    ProbMask,
    TriangularCausalMask,
)


def test_triangular_causal_mask_shape_and_dtype():
    mask = TriangularCausalMask(B=2, L=4)

    assert mask.mask.shape == (2, 1, 4, 4)
    assert mask.mask.dtype == torch.bool


def test_triangular_causal_mask_marks_future_positions():
    mask = TriangularCausalMask(B=1, L=4)

    expected = torch.tensor(
        [
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask.mask[0], expected)


def test_prob_mask_matches_selected_query_rows():
    scores = torch.zeros(1, 1, 2, 4)
    index = torch.tensor([[[1, 3]]])

    mask = ProbMask(B=1, H=1, L=4, index=index, scores=scores)

    expected = torch.tensor(
        [[[[False, False, True, True], [False, False, False, False]]]],
        dtype=torch.bool,
    )
    assert torch.equal(mask.mask, expected)
