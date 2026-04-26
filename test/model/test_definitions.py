from picid.model import definitions as model_definitions


def test_model_definitions_are_flattened_into_all_tasks():
    assert model_definitions.ALL_TASKS == [
        *model_definitions.REGRESSION_TASKS,
        *model_definitions.CLASSIFICATION_TASKS,
        *model_definitions.FORECASTING_TASKS,
        *model_definitions.STATE_FORECASTING_TASKS,
    ]


def test_model_definitions_do_not_repeat_task_names():
    assert len(model_definitions.ALL_TASKS) == len(set(model_definitions.ALL_TASKS))
