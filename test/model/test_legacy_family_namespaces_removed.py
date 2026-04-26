"""Post-migration checks that legacy family namespaces are gone."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "picid.model.methods",
        "picid.model.methods.naive_model",
        "picid.model.wrappers",
        "picid.model.wrappers.base",
        "picid.model.wrappers.naive_model_wrapper",
        "picid.model.wrappers.fit_predict_tabpfn_wrapper",
    ],
)
def test_legacy_family_namespaces_are_removed(module_name):
    with pytest.raises(ModuleNotFoundError):
        import_module(module_name)
