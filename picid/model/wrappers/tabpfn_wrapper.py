"""Legacy shim for the canonical TabPFN wrapper family."""

from picid.model.estimators.tabpfn.wrapper import (
    TabPFNWrapper,
    generate_and_fit_polynomial,
)

__all__ = ["TabPFNWrapper", "generate_and_fit_polynomial"]
