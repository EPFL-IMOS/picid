"""Main entry point for training and testing models."""

import random
import torch
# shutil
from typing import Any
import hydra
import lightning as L
import numpy as np
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from lmetk.data.preprocessing.preprocessor import PreProcessor
from lmetk.utils_refactor import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)



def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="run"
    )
    cfg = hydra.compose(config_name="run.yaml")
    return cfg


torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.datasource._target_}>")
    dataset_source = hydra.utils.instantiate(cfg.datasource, _recursive_=False)

    preprocessor = PreProcessor(dataset_source)
    preprocessor.pipeline()

    #data_lst = dataset_source.get_data()

    # if isinstance(data_lst, list)==False:
    #     # Check that it is the datasource class

    #     # Wrap into list to follow the pipeline
    #     data_lst = [data_lst]

    
    # Finished here for now 
    # data_lst
    
    #dataset_source._pipeline()

    # Splitting

    # Preprocessing / remove trend 
    
    
    # Get data
    features, time_features = dataset_source.get_data()
    scaler = dataset_source.get_scaler()['features']
    
    # Extract additional inputs to dataset if needed.
    add_args = {}
    if cfg.datasource.load_arguments.data_name == 'railway':
        add_args['mask'] = dataset_source.data_dict['mask']

    # Prepare datasets
    dataset_train = hydra.utils.instantiate(
        cfg.dataset,
        features=features["train"], 
        time_features=time_features["train"],
        **add_args
        )

    dataset_val = hydra.utils.instantiate(
        cfg.dataset,
        features=features["val"], 
        time_features=time_features["val"],
        **add_args
        )
    
    dataset_test = hydra.utils.instantiate(
            cfg.dataset,
            features=features["test"], 
            time_features=time_features["test"],
            **add_args
            )
    print(f"Train dataset size: {dataset_train.__len__()}")
    print(f"Validation dataset size: {dataset_val.__len__()}")
    print(f"Test dataset size: {dataset_test.__len__()}")

    # Instantiate dataloader
    datamodule = hydra.utils.instantiate(cfg.datamodule, 
        dataset_train = dataset_train,
        dataset_val = dataset_val,
        dataset_test = dataset_test,
    )

    evaluator = hydra.utils.instantiate(cfg.evaluator, scaler=scaler)
    print(f"Evaluator: {cfg.evaluator._target_}")

    # Instantiate optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    print(f"Optimizer: {cfg.optimizer._target_}")
    
    # Instantiate loss function
    loss = hydra.utils.instantiate(cfg.loss)
    print(f"Loss: {cfg.loss._target_}")

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # TimePFN
    #from lmetk.model.time_pfn.TimePFN import TimePFN
    from lmetk.pipeline.base import LMETKModel

    # train_loader = datamodule.train_dataloader()
    # val_loader = datamodule.val_dataloader()
    # test_loader = datamodule.test_dataloader()

    #seq_x, seq_y, seq_x_mark, seq_y_mark = next(iter(train_loader))
    
    
    # backbone = TimePFN(cfg.model)
    # backbone.load_state_dict(torch.load(cfg.model.load_path))
    add_backbone_args = {}
    if cfg.datasource.load_arguments.data_name == 'railway':
        add_backbone_args['train_dataloader'] = datamodule.train_dataloader()
        add_backbone_args['val_dataloader'] = datamodule.val_dataloader()

    backbone_wrapper = hydra.utils.instantiate(cfg.model, **add_backbone_args) # TimePFNWrapper(backbone, **cfg.model)
    
    if cfg.model.get('load_path', False):
        backbone_wrapper.backbone.load_state_dict(torch.load(cfg.model.load_path))
    
    model = LMETKModel(
        #backbone=backbone, 
        #backbone_wrapper=backbone_wrapper,
        backbone = backbone_wrapper,
        loss=loss,
        evaluator=evaluator,
        optimizer=optimizer,
        )
    
    
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
    )

    # if cfg.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(
    #         model=model, datamodule=datamodule,
    #     )
    
    trainer.test(model=model, datamodule=datamodule,)
    

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }
    
    
    # ----REFACTOR 2---- 29/05 evening
    
    # device='cuda:0'
    # model.to(device)
    # #preds = model.forward(x_enc=seq_x, x_mark_enc=seq_x_mark, x_dec=seq_y, x_mark_dec=seq_y_mark)

    # preds = []
    # trues = []
    # folder_path = './test_results/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # model.eval()
    # with torch.no_grad():
    #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #         batch_x = batch_x.float().to(device)
    #         batch_y = batch_y.float().to(device)
    #         channel_size = batch_x.size(2)
    #         max_channel_size = cfg.model.max_channel
    #         number_of_iter = channel_size // max_channel_size + 1

    #         pred = None
    #         true = None

    #         for j in range(number_of_iter):
    #             batch_x_subset = batch_x[:, :, j * max_channel_size:(j + 1) * max_channel_size]
    #             batch_y_subset = batch_y[:, :, j * max_channel_size:(j + 1) * max_channel_size]

    #             # if 'PEMS' in cfg.model.data or 'Solar' in cfg.model.data:
    #             #     batch_x_mark = None
    #             #     batch_y_mark = None
    #             # else:
    #             batch_x_mark = batch_x_mark.to(device)
    #             batch_y_mark = batch_y_mark.to(device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y_subset[:, -cfg.model.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y_subset[:, :cfg.model.label_len, :], dec_inp], dim=1).float().to(device)
    #             # encoder - decoder
    #             if cfg.model.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if cfg.model.output_attention:
    #                         outputs = model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if cfg.model.output_attention:
    #                     outputs = model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]

    #                 else:
    #                     outputs = model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)

    #             f_dim = -1 if cfg.datasource.loader.load_arguments.features_mode == 'MS' else 0

    #             outputs = outputs[:, -cfg.model.pred_len:, f_dim:]
    #             batch_y_subset = batch_y_subset[:, -cfg.model.pred_len:, f_dim:].to(device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y_subset = batch_y_subset.detach().cpu().numpy()

    #             # # Apply inverse transform if needed
    #             # if scaler and cfg.model.inverse:
    #             #     shape_pred = outputs.shape
    #             #     shape_true = batch_y_subset.shape
    #             #     outputs = scaler.inverse_transform(outputs.squeeze(0)).reshape(shape_pred)
    #             #     batch_y_subset = scaler.inverse_transform(batch_y_subset.squeeze(0)).reshape(shape_true)

    #             # Update evaluator with current batch (replaces the accumulation logic)
    #             model_out = {
    #                 'predictions': outputs,
    #                 'targets': batch_y_subset
    #             }
    #             evaluator.update(model_out)
                
                

    #     # Compute final metrics (replaces the old metric computation)
    #     final_metrics = evaluator.compute()

        

    #     # If you specifically need MAE and MSE for backward compatibility:
    #     mae_value = final_metrics.get('mae', 0)
    #     mse_value = final_metrics.get('mse', 0)
    #     print(f'mse:{mse_value:.6f}, mae:{mae_value:.6f}')

            
            
            
            
            
            
            
            
            
           # ----REFACTOR 1---- 29/05 midday 
            
            # # Optional: Access visualization data if needed
            # if visualization_data:
            #     print(f"Stored {len(visualization_data)} visualization samples")

    # Commented
    #             outputs = outputs[:, -cfg.model.pred_len:, f_dim:]
    #             batch_y_subset = batch_y_subset[:, -cfg.model.pred_len:, f_dim:].to(device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y_subset = batch_y_subset.detach().cpu().numpy()

    #             if pred is None:
    #                 pred = outputs
    #                 true = batch_y_subset
    #             else:
    #                 pred = np.concatenate((pred, outputs), axis=2)
    #                 true = np.concatenate((true, batch_y_subset), axis=2)
            
            
            
            
    #         #true = batch_y.detach().cpu().numpy()
    #         if true.shape != pred.shape:
    #             print(true.shape, pred.shape)
            
    #         if scaler and cfg.model.inverse:
    #             shape = pred.shape
    #             pred = scaler.inverse_transform(pred.squeeze(0)).reshape(shape)
    #             true = scaler.inverse_transform(true.squeeze(0)).reshape(shape)

            
    #         preds.append(pred)
    #         trues.append(true)
    #         if i % 20 == 0:
    #             input = batch_x.detach().cpu().numpy()
    #             if scaler and cfg.model.inverse:
    #                 shape = input.shape
    #                 input = scaler.inverse_transform(input.squeeze(0)).reshape(shape)
    #             gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #             pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #             #visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    # preds = np.array(preds)
    # trues = np.array(trues)
    # print('test shape:', preds.shape, trues.shape)
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)
    # #self.record_test_shape = preds.shape

    # # result save
    # folder_path = f'{os.getcwd()}/results/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # mae, mse, rmse, mape, mspe = metric(preds, trues)
    # print('mse:{}, mae:{}'.format(mse, mae))
    # f = open("result_long_term_forecast.txt", 'a')
    # f.write("  \n")
    # f.write('mse:{}, mae:{}'.format(mse, mae))
    # f.write('\n')
    # f.write('\n')
    # f.close()

    

    
    


    # model: LightningModule = hydra.utils.instantiate(
    #     cfg.model,
    #     evaluator=cfg.evaluator,
    #     optimizer=cfg.optimizer,
    #     loss=cfg.loss,
    # )
    

    # # Preprocess dataset and load the splits
    # log.info("Instantiating preprocessor...")
    # transform_config = cfg.get("transforms", None)
    # preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    # dataset_train, dataset_val, dataset_test = (
    #     preprocessor.load_dataset_splits(cfg.dataset.split_params)
    # )
    # # Prepare datamodule
    # log.info("Instantiating datamodule...")
    # if cfg.dataset.parameters.task_level in ["node", "graph"]:
    #     datamodule = TBDataloader(
    #         dataset_train=dataset_train,
    #         dataset_val=dataset_val,
    #         dataset_test=dataset_test,
    #         **cfg.dataset.get("dataloader_params", {}),
    #     )
    # else:
    #     raise ValueError("Invalid task_level")

    # # Model for us is Network + logic: inputs backbone, readout, losses
    # log.info(f"Instantiating model <{cfg.model._target_}>")
    # model: LightningModule = hydra.utils.instantiate(
    #     cfg.model,
    #     evaluator=cfg.evaluator,
    #     optimizer=cfg.optimizer,
    #     loss=cfg.loss,
    # )

    # log.info("Instantiating callbacks...")
    # callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # log.info("Instantiating loggers...")
    # logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # # Log to wandb preprocessor time
    # if logger:
    #     for log_temp in logger:
    #         if isinstance(log_temp, L.pytorch.loggers.wandb.WandbLogger):
    #             log_temp.log_metrics(
    #                 {
    #                     "preprocessor_time": preprocessor.preprocessing_time,
    #                 }
    #             )

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(
    #     cfg.trainer,
    #     callbacks=callbacks,
    #     logger=logger,
    #     num_sanity_val_steps=0,
    #     log_every_n_steps=0,  # Avoid console logging
    # )

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }

    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)

    # if cfg.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(
    #         model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    #     )

    # train_metrics = trainer.callback_metrics
    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     test_best_model_path = True
    #     if cfg.get("ckpt_path"):
    #         ckpt_path = cfg.ckpt_path
    #         log.info(
    #             f"Attempting to load weights from the provided ckpt_path: {ckpt_path}"
    #         )
    #         try:
    #             trainer.test(
    #                 model=model, datamodule=datamodule, ckpt_path=ckpt_path
    #             )
    #             test_best_model_path = False  # do not test "best model" if a valid ckpt_path is provided
    #         except FileNotFoundError:
    #             log.warning(
    #                 f"No checkpoint file found at the provided ckpt_path: {ckpt_path}."
    #             )
    #             log.info("Trying with best model instead...")
    #     if test_best_model_path:
    #         ckpt_path = trainer.checkpoint_callback.best_model_path
    #         if ckpt_path == "":
    #             log.warning(
    #                 "Best ckpt not found! Using current weights for testing..."
    #             )
    #             ckpt_path = None
    #         trainer.test(
    #             model=model, datamodule=datamodule, ckpt_path=ckpt_path
    #         )

    #     # Remove saved model (useful with sweeps)
    #     # Check if the file exists and delete it
    #     if os.path.exists(ckpt_path):
    #         os.remove(ckpt_path)

    # test_metrics = trainer.callback_metrics

    # # Merge train and test metrics
    # metric_dict = {**train_metrics, **test_metrics}

    return None, None # metric_dict, object_dict


def count_number_of_parameters(
    model: torch.nn.Module, only_trainable: bool = True
) -> int:
    """Count the number of trainable params.

    If all params, specify only_trainable = False.

    Ref:
        - https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=brando_miranda
        - https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    only_trainable : bool, optional
        If True, only count trainable parameters (default: True).

    Returns
    -------
    int
        The number of parameters.
    """
    if only_trainable:
        num_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:  # counts trainable and none-traibale
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    assert num_params > 0, f"Err: {num_params=}"
    return int(num_params)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    float | None
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
