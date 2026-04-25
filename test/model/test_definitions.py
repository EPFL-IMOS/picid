from picid.baselines import definitions as legacy_definitions
from picid.model import definitions as model_definitions


def test_model_definitions_match_legacy_definitions():
    assert model_definitions.REGRESSION_TASKS == legacy_definitions.REGRESSION_TASKS
    assert (
        model_definitions.CLASSIFICATION_TASKS
        == legacy_definitions.CLASSIFICATION_TASKS
    )
    assert model_definitions.FORECASTING_TASKS == legacy_definitions.FORECASTING_TASKS
    assert (
        model_definitions.STATE_FORECASTING_TASKS
        == legacy_definitions.STATE_FORECASTING_TASKS
    )
    assert model_definitions.ALL_TASKS == legacy_definitions.ALL_TASKS


def test_legacy_definitions_reexport_model_definitions():
    assert legacy_definitions.REGRESSION_TASKS is model_definitions.REGRESSION_TASKS
    assert (
        legacy_definitions.CLASSIFICATION_TASKS
        is model_definitions.CLASSIFICATION_TASKS
    )
    assert legacy_definitions.FORECASTING_TASKS is model_definitions.FORECASTING_TASKS
    assert (
        legacy_definitions.STATE_FORECASTING_TASKS
        is model_definitions.STATE_FORECASTING_TASKS
    )
    assert legacy_definitions.ALL_TASKS is model_definitions.ALL_TASKS
