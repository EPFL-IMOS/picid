"""Minimal GraphBench-like entry point that hides Hydra details."""

from picid.run import *

from picid.simplified_run_objects import (
    Experiment,
    Loader,
    Transforms,
    Model,
    Evaluator,
)


def estimate_loader_lengths_REMOVE_THIS(datamodule):
    # Log the sizes of one batch for each data loader
    loader_lengths = {}
    for split, loader in zip(
        ["train", "val", "test"],
        [
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader(),
        ],
    ):
        loader_lengths[split] = len(loader)

    return loader_lengths


# Example usage: mirrors graphbench pattern
if __name__ == "__main__":

    exp = Experiment(
        [
            "paths=rt_local",
            "experiment=nb14/prognostics/raw/stf",
            "debug=nox_testing_fit_predict",
            "num_threads=1",
            "seed=50",
            "trainer.max_epochs=1",
        ]
    )

    exp.cfg.paths.output_dir = ".artifacts/simplified_run"

    transforms = Transforms.build(exp.cfg)
    print("Transforms:", list(transforms.names()))

    # transforms and loading needs to be split long term.
    data, datamodule = Loader(exp.cfg, transforms).load()

    evaluators = Evaluator(exp.cfg).build()

    ############################################
    # THIS MUST FOR SURE GO IN THE FINAL SCRIPT
    # TODO: why here? why not at the top of the file?
    # This allows configs to infer data dims from data
    # by using a resolver like ${infer_data_dim:input_key,dim}
    register_data_dim_resolver(data=data["train"])
    register_infer_dataloader_length_resolver(
        estimate_loader_lengths_REMOVE_THIS(datamodule)
    )
    ############################################

    metrics, _ = Model(exp.cfg).optimize(datamodule, evaluators)
    print("Metrics:", metrics)
