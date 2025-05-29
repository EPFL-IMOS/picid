from pathlib import Path
import pytest
import hydra

from box import Box
import torch

from lmetk.data.datamodules.datamodule import BaseDataModule

# This import is necessary to register the plugin
from test.utils import ProjectSearchPathPlugin
ProjectSearchPathPlugin.register()


@pytest.fixture(scope="module", autouse=True)
def hydra_context(request):
    with hydra.initialize(version_base="1.3", config_path="configs", job_name="run"):
        yield


@pytest.fixture(scope="module")
def cmapss_datamodule():
    cfg = hydra.compose(
        config_name="test.yaml",
        overrides=[
            "+datamodule=test_simplified_rul",
            "dataset=concept_rul_dataset",
            "+datasource=concepts_cmapss",
            "paths=rt_local",
        ],
        return_hydra_config=True,
    )

    print(f"Config file: {cfg.dataset}")

    data_source = hydra.utils.instantiate(cfg.datasource.loader)

    data_source._load_data()
    data_source._generate_splits()
    data_source._apply_transformations()

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_train = hydra.utils.instantiate(cfg.dataset, data_split=in_box)

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_val = hydra.utils.instantiate(cfg.dataset, data_split=in_box)

    in_box = Box({k: v.train for k, v in data_source.data_dict.items()})
    ds_test = hydra.utils.instantiate(cfg.dataset, data_split=in_box)

    sample = next(iter(ds_train))
    for id_t, t in enumerate(sample):
        print(f"Tensor {id_t} shape: {t.shape}")

    datasets = Box(
        dict(
            train=ds_train,
            val=ds_val,
            test=ds_test,
        )
    )

    # Use hydra to partially initialize BaseDataModule, override datasets with our datasets
    dm = hydra.utils.instantiate(cfg.datamodule, datasets=datasets)

    return dm


def test_batch_contents(cmapss_datamodule):
    """Test the contents of a batch from the datamodule.
    Fetch a batch from the train dataloader and check its contents.
    Verify with a batch that is stored on the file system.
    """

    path = Path("test/assets/data/datasets")
    batch_file = path / "reference_batch_cmapss.pt"
    batch = next(iter(cmapss_datamodule.train_dataloader()))

    if batch_file.exists():
        reference = torch.load(batch_file)
        # Compare each tensor in the batch with the reference
        for idx, (b, r) in enumerate(zip(batch, reference)):
            assert torch.allclose(b, r), f"Batch tensor {idx} does not match reference"
    else:
        # Uncomment the following line to recreate the reference batch
        # torch.save(batch, batch_file)
        assert False, (
            f"Reference batch file {batch_file} does not exist. Uncomment save line to create it."
        )


def test_cmapss_datamodule(cmapss_datamodule):
    assert cmapss_datamodule is not None, "Datamodule is None"
    assert isinstance(cmapss_datamodule, BaseDataModule), (
        "Datamodule is not of the expected type"
    )
    assert len(cmapss_datamodule.train_dataloader()) > 0, "Train dataloader is empty"
    assert len(cmapss_datamodule.val_dataloader()) > 0, "Validation dataloader is empty"
    assert len(cmapss_datamodule.test_dataloader()) > 0, "Test dataloader is empty"


def test_sample_shapes_from_train_dataset(cmapss_datamodule):
    ds = cmapss_datamodule.datasets.train
    sample = next(iter(ds))
    assert len(sample) == 3, "Sample does not contain the expected number of tensors"
    assert sample[0].shape == torch.Size([24]), (
        f"Tensor 0 shape is incorrect: {sample[0].shape}"
    )
    assert sample[1].shape == torch.Size([]), (
        f"Tensor 1 shape is incorrect: {sample[1].shape}"
    )
    assert sample[2].shape == torch.Size([2]), (
        f"Tensor 2 shape is incorrect: {sample[2].shape}"
    )


def test_sample_shapes_from_val_dataset(cmapss_datamodule):
    ds = cmapss_datamodule.datasets.val
    sample = next(iter(ds))
    assert len(sample) == 3, "Sample does not contain the expected number of tensors"
    assert sample[0].shape == torch.Size([24]), (
        f"Tensor 0 shape is incorrect: {sample[0].shape}"
    )
    assert sample[1].shape == torch.Size([]), (
        f"Tensor 1 shape is incorrect: {sample[1].shape}"
    )
    assert sample[2].shape == torch.Size([2]), (
        f"Tensor 2 shape is incorrect: {sample[2].shape}"
    )


def test_sample_shapes_from_test_dataset(cmapss_datamodule):
    ds = cmapss_datamodule.datasets.test
    sample = next(iter(ds))
    assert len(sample) == 3, "Sample does not contain the expected number of tensors"
    assert sample[0].shape == torch.Size([24]), (
        f"Tensor 0 shape is incorrect: {sample[0].shape}"
    )
    assert sample[1].shape == torch.Size([]), (
        f"Tensor 1 shape is incorrect: {sample[1].shape}"
    )
    assert sample[2].shape == torch.Size([2]), (
        f"Tensor 2 shape is incorrect: {sample[2].shape}"
    )
