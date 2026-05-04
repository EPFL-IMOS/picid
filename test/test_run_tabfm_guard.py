"""Tests for the tabfm dependency-group guard in picid.model.tabfm_guard."""

import importlib.util

import pytest
from omegaconf import OmegaConf

from picid.model.tabfm_guard import TABFM_REQUIREMENTS as _TABFM_REQUIREMENTS
from picid.model.tabfm_guard import check_tabfm_available as _check_tabfm_available


def _cfg(target: str):
    return OmegaConf.create({"model": {"_target_": target}})


# ---------------------------------------------------------------------------
# Guard passes for non-tabfm models
# ---------------------------------------------------------------------------


def test_non_tabfm_target_raises_nothing():
    _check_tabfm_available(_cfg("picid.model.estimators.cnn.CNNWrapper"))


def test_empty_target_raises_nothing():
    _check_tabfm_available(OmegaConf.create({}))


def test_no_model_key_raises_nothing():
    _check_tabfm_available(OmegaConf.create({"trainer": {}}))


# ---------------------------------------------------------------------------
# Guard raises when package is missing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_prefix, import_name, display_name", [
    (prefix, imp, disp)
    for prefix, (imp, disp) in _TABFM_REQUIREMENTS.items()
])
def test_missing_package_raises_runtime_error(
    monkeypatch, module_prefix, import_name, display_name
):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match="uv sync --group tabfm"):
        _check_tabfm_available(_cfg(f"{module_prefix}.wrapper.SomeWrapper"))


@pytest.mark.parametrize("module_prefix, import_name, display_name", [
    (prefix, imp, disp)
    for prefix, (imp, disp) in _TABFM_REQUIREMENTS.items()
])
def test_error_message_names_the_missing_package(
    monkeypatch, module_prefix, import_name, display_name
):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match=display_name):
        _check_tabfm_available(_cfg(f"{module_prefix}.wrapper.SomeWrapper"))


# ---------------------------------------------------------------------------
# Guard passes when package is present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_prefix, import_name, display_name", [
    (prefix, imp, disp)
    for prefix, (imp, disp) in _TABFM_REQUIREMENTS.items()
])
def test_installed_package_does_not_raise(
    monkeypatch, module_prefix, import_name, display_name
):
    fake_spec = object()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: fake_spec)

    _check_tabfm_available(_cfg(f"{module_prefix}.wrapper.SomeWrapper"))


# ---------------------------------------------------------------------------
# Only the matched package is checked
# ---------------------------------------------------------------------------


def test_only_matching_package_is_checked(monkeypatch):
    checked = []

    def _fake_find_spec(name):
        checked.append(name)
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    _check_tabfm_available(_cfg("picid.model.estimators.tabpfn.wrapper.FitPredictTabPFNWrapper"))

    assert checked == ["tabpfn"]
