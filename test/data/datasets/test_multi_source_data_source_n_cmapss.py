
# This import is necessary to register the plugin
from test.utils import ProjectSearchPathPlugin

import hydra
import pytest
from omegaconf import OmegaConf

import lmetk
from lmetk.data.preprocessing.multi_source_preprocessor import MultiSourcePreProcessor
from lmetk.transforms.base.transform_manager import ConfigTransformManager

ProjectSearchPathPlugin.register()


@pytest.fixture(scope="module", autouse=True)
def hydra_context(request):
    with hydra.initialize(version_base="1.3", config_path="configs", job_name="run"):
        yield


@pytest.fixture(scope="module")
def path_cfg(request) -> str:
    """Return the path to the configuration file."""
    return request.config.getoption("--paths")


@pytest.fixture(scope="module")
def multi_source_data_source(path_cfg):

    cfg = hydra.compose(
        config_name="test.yaml",
        overrides=[
            "+datamodule=test_simplified_rul",
            "dataset=concept_rul_multiunit_dataset",
            "datasource=concepts_n_cmapss_multi_source",
            f"paths={path_cfg}",
        ],
        return_hydra_config=True,
    )

    # print(f"Config file: {cfg.dataset}")

    print(OmegaConf.to_yaml(cfg.datasource, resolve=True))

    dataset_source = hydra.utils.instantiate(cfg.datasource)

    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)

    preprocessor = MultiSourcePreProcessor(
        dataset_source, transforms_manager, logger=log
    )
    preprocessor.pipeline()

    data_dict = preprocessor.get_processed_data()

    dataset_train = hydra.utils.instantiate(
        cfg.dataset, data_dict=data_dict, split="train", _recursive_=False
    )
    dataset_val = hydra.utils.instantiate(
        cfg.dataset, data_dict=data_dict, split="val", _recursive_=False
    )
    dataset_test = hydra.utils.instantiate(
        cfg.dataset, data_dict=data_dict, split="test", _recursive_=False
    )

    # data_source = hydra.utils.instantiate(cfg.datasource)

    # data_source._pipeline()

    # in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    # ds_train = hydra.utils.instantiate(cfg.dataset, data_split=in_box, _recursive_=False)

    # in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    # # To partially initialize without instantiating sub _target_ fields, use _recursive_=False
    # ds_val = hydra.utils.instantiate(cfg.dataset, data_split=in_box, _recursive_=False)

    # in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    # ds_test = hydra.utils.instantiate(cfg.dataset, data_split=in_box, _recursive_=False)

    sample = next(iter(dataset_train))
    for id_t, t in enumerate(sample):
        print(f"Tensor {id_t} shape: {t.shape}")

    return dataset_source


def test_create_multi_source_data_source(multi_source_data_source):
    assert isinstance(
        multi_source_data_source,
        lmetk.data.datasources.multi_source_pipeline.MultiSource,
    )
    assert (
        len(multi_source_data_source.data_lst) > 0
    ), "No sources found in the multi-source pipeline"

    # Check if data_dict is populated
    assert multi_source_data_source.data_lst[0], "Data dictionary is empty"

    # Check if scalers are instantiated
    # assert multi_source_data_source.get_scaler(), "Scalers are not instantiated"

    # Check if data is split correctly
    assert (
        "train" in multi_source_data_source.data_lst["features"]
    ), "Train split not found in features"

    # TODO: Uncomment when validation split is available, why is it not available?
    # assert 'val' in multi_source_data_source.data_dict['features'], "Validation split not found in features"
    assert (
        "test" in multi_source_data_source.data_lst["features"]
    ), "Test split not found in features"

    assert (
        "test" in multi_source_data_source.data_lst["features"]
    ), "Test split not found in features"
