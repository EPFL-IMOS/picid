"""Legacy shim for the canonical statistical backbones."""

from picid.model.estimators.statistical.model import ExponentialBaseline
from picid.model.estimators.statistical.model import LinearBaseline
from picid.model.estimators.statistical.model import PolynomialBaseline

__all__ = ["ExponentialBaseline", "LinearBaseline", "PolynomialBaseline"]
