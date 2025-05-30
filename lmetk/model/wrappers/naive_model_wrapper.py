"""Wrapper for the NaiveModel model."""
from .base import AbstractWrapper
from lmetk.model.methods.naive_model import NaiveModelBaseline
from lightning_fabric.utilities.data import AttributeDict

class NaiveModelWrapper(AbstractWrapper):
    r"""Wrapper for the NaiveModel model.

    This wrapper defines the forward pass of the NaiveModel model.
    """
    def __init__(self, **kwargs):
        backbone=NaiveModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        batch_x = batch['batch_x']
        batch_y = batch['batch_y']
        
        # batch_x_mark = batch['batch_x_mark']
        # batch_y_mark = batch['batch_y_mark']
        

        outputs = self.backbone(batch_x)
            
        # I believe the logic of processing outputs should be disentangled from the model wrapper into some additional module. 
        f_dim = -1 if self.kwargs.features_mode == 'MS' else 0

        outputs = outputs[:, -self.kwargs.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.kwargs.pred_len:, f_dim:]
        
        outputs = outputs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
            
        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {
            'predictions': outputs,
            'targets': batch_y
        }
        
        return model_out