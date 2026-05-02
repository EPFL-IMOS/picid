"""Tests for picid.interface.model.wrapper module.

Coverage target: >=95% of picid/interface/model/wrapper.py

Tests verify the ModelWrapper's forward-pass composition (pre→model→post),
attribute delegation, None-function defaults, and gradient flow.
"""

import pytest
import torch
from torch import nn

from picid.interface.model.wrapper import ModelWrapper, empty_function


class _DoubleModule(nn.Module):
    """Minimal model that doubles its input."""

    def forward(self, x):
        return x * 2


class _LinearModule(nn.Module):
    """Wraps a single Linear layer for gradient tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
class TestEmptyFunction:
    """Tests for the identity function used as default processor."""

    def test_returns_same_object(self):
        """empty_function returns its argument unchanged.

        **Methodology**: Pass various objects.

        **Expected**: Same object (identity) returned.
        """
        tensor = torch.tensor([1.0])
        assert empty_function(tensor) is tensor
        assert empty_function(42) == 42
        assert empty_function("hello") == "hello"


@pytest.mark.unit
class TestModelWrapper:
    """Tests for ModelWrapper — transparent pre/post-processing wrapper."""

    def test_init_defaults_to_identity(self):
        """No pre/post processors defaults to empty_function.

        **Methodology**: Create wrapper without processors.

        **Expected**: Both processors are empty_function.
        """
        model = _DoubleModule()
        wrapper = ModelWrapper(model=model)
        assert wrapper._pre_process_function is empty_function
        assert wrapper._post_process_function is empty_function

    def test_init_with_custom_processors(self):
        """Custom pre and post processors are stored correctly.

        **Methodology**: Supply lambdas for both.

        **Expected**: Stored functions match supplied ones.
        """

        def pre(x):
            return x + 1

        def post(x):
            return x - 1

        wrapper = ModelWrapper(
            model=_DoubleModule(), pre_process_function=pre, post_process_function=post
        )
        assert wrapper._pre_process_function is pre
        assert wrapper._post_process_function is post

    def test_init_only_pre_processor(self):
        """Supplying only pre_process leaves post as identity.

        **Expected**: post is empty_function.
        """

        def pre(x):
            return x + 1

        wrapper = ModelWrapper(model=_DoubleModule(), pre_process_function=pre)
        assert wrapper._pre_process_function is pre
        assert wrapper._post_process_function is empty_function

    def test_init_only_post_processor(self):
        """Supplying only post_process leaves pre as identity.

        **Expected**: pre is empty_function.
        """

        def post(x):
            return x * 3

        wrapper = ModelWrapper(model=_DoubleModule(), post_process_function=post)
        assert wrapper._pre_process_function is empty_function
        assert wrapper._post_process_function is post

    def test_forward_no_processors(self):
        """Forward pass without processors applies only the model.

        **Methodology**: Input [1, 2] → model doubles → [2, 4].

        **Expected**: Output equals input * 2.
        """
        wrapper = ModelWrapper(model=_DoubleModule())
        x = torch.tensor([1.0, 2.0])
        result = wrapper(x)
        torch.testing.assert_close(result, torch.tensor([2.0, 4.0]))

    def test_forward_with_pre_processor(self):
        """Pre-processor is applied before the model.

        **Methodology**: pre adds 10, model doubles → (1+10)*2 = 22.

        **Expected**: Output is 22.
        """
        wrapper = ModelWrapper(
            model=_DoubleModule(),
            pre_process_function=lambda x: x + 10,
        )
        result = wrapper(torch.tensor([1.0]))
        torch.testing.assert_close(result, torch.tensor([22.0]))

    def test_forward_with_post_processor(self):
        """Post-processor is applied after the model.

        **Methodology**: model doubles, post adds 100 → 1*2 + 100 = 102.

        **Expected**: Output is 102.
        """
        wrapper = ModelWrapper(
            model=_DoubleModule(),
            post_process_function=lambda x: x + 100,
        )
        result = wrapper(torch.tensor([1.0]))
        torch.testing.assert_close(result, torch.tensor([102.0]))

    def test_forward_both_processors_correct_order(self):
        """Pre → model → post order is preserved.

        **Methodology**: pre adds 1, model doubles, post triples.
        Input 2 → pre: 3 → model: 6 → post: 18.

        **Expected**: Output is 18.
        """
        wrapper = ModelWrapper(
            model=_DoubleModule(),
            pre_process_function=lambda x: x + 1,
            post_process_function=lambda x: x * 3,
        )
        result = wrapper(torch.tensor([2.0]))
        torch.testing.assert_close(result, torch.tensor([18.0]))

    def test_forward_extra_kwargs_ignored(self):
        """Extra keyword arguments do not raise.

        **Methodology**: Pass extra_kwarg=True.

        **Expected**: No error, output correct.
        """
        wrapper = ModelWrapper(model=_DoubleModule())
        result = wrapper(torch.tensor([1.0]), some_kwarg=True)
        torch.testing.assert_close(result, torch.tensor([2.0]))

    def test_getattr_delegates_to_base_model(self):
        """Unknown attributes are delegated to base_model.

        **Methodology**: Set custom_attr on base_model, access via wrapper.

        **Expected**: wrapper.custom_attr == 42.
        """
        model = _DoubleModule()
        model.custom_attr = 42
        wrapper = ModelWrapper(model=model)
        assert wrapper.custom_attr == 42

    def test_getattr_raises_for_missing(self):
        """Truly missing attribute raises AttributeError.

        **Methodology**: Access nonexistent attribute.

        **Expected**: AttributeError raised.
        """
        wrapper = ModelWrapper(model=_DoubleModule())
        with pytest.raises(AttributeError):
            _ = wrapper.totally_nonexistent_attr

    def test_gradients_flow_through_wrapper(self):
        """Gradients flow through the wrapper to the base model.

        **Methodology**: Wrap a Linear layer, forward, backward, check grad.

        **Expected**: Linear weight gradient is not None.
        """
        model = _LinearModule()
        wrapper = ModelWrapper(model=model)
        x = torch.randn(1, 4)
        out = wrapper(x)
        loss = out.sum()
        loss.backward()
        assert model.linear.weight.grad is not None

    def test_is_nn_module(self):
        """ModelWrapper is an nn.Module instance.

        **Expected**: isinstance check passes.
        """
        wrapper = ModelWrapper(model=_DoubleModule())
        assert isinstance(wrapper, nn.Module)

    def test_parameters_include_base_model(self):
        """Wrapper's parameters() includes base_model parameters.

        **Methodology**: Wrap a Linear layer, count parameters.

        **Expected**: At least one parameter (the Linear weight).
        """
        model = _LinearModule()
        wrapper = ModelWrapper(model=model)
        params = list(wrapper.parameters())
        assert len(params) >= 1
