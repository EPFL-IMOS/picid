
import os
import pytest
import hydra
from pathlib import Path
from lmetk.data.datasets.multiunit_datasource import MultiUnitDataSource


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins

from box import Box
import hydra

from lmetk.data.datasets.rul_datamodule import ConceptRULDataset, SimlifiedRulDataModule

class ProjectSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        search_path.append(
            provider="lmetk.defaults" , path=f"file://{Path.cwd()}/configs"
        )

    def register() -> None:

        """Hydra users should call this function before invoking @hydra.main"""
        print("CWD:", Path.cwd())
        os.environ["PROJECT_ROOT"] = str(Path.cwd())
        Plugins.instance().register(ProjectSearchPathPlugin)

ProjectSearchPathPlugin.register()



@pytest.fixture(scope="module", autouse=True)
def hydra_context(request):

    with hydra.initialize(version_base="1.3", config_path="configs", job_name="run"):
        yield

def test_dm():
    data_name = 'cmapss_refactored'

    cfg = hydra.compose(
        config_name="test.yaml",
        overrides=[f'dataset={data_name}', f"paths=rt_local"], 
        return_hydra_config=True,
    )

    print(f"Config file: {cfg.dataset.loader}")

    obj = hydra.utils.instantiate(cfg.dataset.loader, _convert_='all')
    data_source = hydra.utils.instantiate(cfg.dataset.loader)

    dd = data_source._load_data()
    data_source._generate_splits()
    data_source._apply_transformations()

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_train = ConceptRULDataset(in_box, window_size=1, stride=1,)

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_val = ConceptRULDataset(in_box, window_size=1, stride=1,)

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_test = ConceptRULDataset(in_box, window_size=1, stride=1,)


    sample = next(iter(ds_train))  
    for id_t, t in enumerate(sample):
        print(f"Tensor {id_t} shape: {t.shape}")


    datasets = Box(dict(
        train=ds_train,
        val=ds_val,
        test=ds_test,
    ))

    dm = SimlifiedRulDataModule(datasets=datasets, batch_size=32)

    sample = next(iter(dm.train_dataloader()))  
    for id_t, t in enumerate(sample):
        print(f"Tensor {id_t} shape: {t.shape}")

# def test_multi_unit_datasource():
#     data_name = 'cmapss_refactored'
#     parameters = hydra.compose(
#         config_name='run.yaml',
#         overrides=[f'dataset={data_name}', 'paths=rt_local'],
#         return_hydra_config=True,
#     )

#     src1 = hydra.utils.instantiate(parameters.dataset.loader)
#     src2 = hydra.utils.instantiate(parameters.dataset.loader)

#     load_args = parameters.dataset.loader
#     splitter = src1.data_splitter
#     scaler = src1.scaler

#     multi_src = MultiUnitDataSource(
#         sources=[src1, src2],
#         load_arguments=load_args,
#         data_splitter=splitter,
#         scaler=scaler,
#     )

#     data_dict, timestamps_dict = multi_src.get_data()

#     for split in ('train', 'val', 'test'):
#         assert split in data_dict
#         assert split in timestamps_dict
#         len_single = len(src1._load_data()[0])
#         assert len(data_dict[split]) == len_single * 2 // 3

#     assert multi_src.get_scaler() is not None