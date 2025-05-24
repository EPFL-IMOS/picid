
import pytest
import hydra

from box import Box
import hydra

from lmetk.data.datasets.rul_datamodule import ConceptRULDataset, SimlifiedRulDataModule

# This import is necessary to register the plugin
from test.utils import ProjectSearchPathPlugin


@pytest.fixture(scope="module", autouse=True)
def hydra_context(request):

    with hydra.initialize(version_base="1.3", config_path="configs", job_name="run"):
        yield

def test_cmapss_loader():
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