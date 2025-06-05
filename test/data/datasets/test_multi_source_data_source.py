from omegaconf import OmegaConf
import pytest
import hydra
import lmetk

from box import Box


# This import is necessary to register the plugin
from test.utils import ProjectSearchPathPlugin
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
            "+datasource=concepts_cmapss_multi_source",
            f"paths={path_cfg}",
        ],
        return_hydra_config=True,
    )

    print(f"Config file: {cfg.dataset}")

    print(OmegaConf.to_yaml(cfg.datasource, resolve=True))

    data_source = hydra.utils.instantiate(cfg.datasource)

    data_source._pipeline()


    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_train = hydra.utils.instantiate(cfg.dataset, data_split=in_box, _recursive_=False)

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    # To partially initialize without instantiating sub _target_ fields, use _recursive_=False
    ds_val = hydra.utils.instantiate(cfg.dataset, data_split=in_box, _recursive_=False)

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_test = hydra.utils.instantiate(cfg.dataset, data_split=in_box, _recursive_=False)

    sample = next(iter(ds_train))
    for id_t, t in enumerate(sample):
        print(f"Tensor {id_t} shape: {t.shape}")

    return data_source


def test_create_multi_source_data_source(multi_source_data_source):
    assert isinstance(multi_source_data_source, lmetk.data.datasources.multi_source_pipeline.MultiSourcePipeline)
    assert len(multi_source_data_source.sources) > 0, "No sources found in the multi-source pipeline"
    
    # check if sources are different units
    units = set()
    for source in multi_source_data_source.sources:
        units.add(source.load_arguments.n_DS)
    
    assert len(units) == len(multi_source_data_source.sources), "All sources are from different units."

    # Check if data_dict is populated
    assert multi_source_data_source.data_dict, "Data dictionary is empty"
    
    # Check if scalers are instantiated
    # assert multi_source_data_source.get_scaler(), "Scalers are not instantiated"
    
    # Check if data is split correctly
    assert 'train' in multi_source_data_source.data_dict['features'], "Train split not found in features"
    assert 'val' in multi_source_data_source.data_dict['features'], "Validation split not found in features"
    assert 'test' in multi_source_data_source.data_dict['features'], "Test split not found in features"

