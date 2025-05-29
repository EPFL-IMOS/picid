"""Wrapper for the TopoTune model."""
import torch
import numpy as np
from .base import AbstractWrapper


class TimePFNWrapper(AbstractWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TopoTune model.
    """

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        batch_x = batch['batch_x']
        batch_y = batch['batch_y']
        batch_x_mark = batch['batch_x_mark']
        batch_y_mark = batch['batch_y_mark']

        channel_size = batch_x.size(2)
        max_channel_size = 161
        number_of_iter = channel_size // max_channel_size + 1

        pred = None
        true = None

        for j in range(number_of_iter):
            batch_x_subset = batch_x[:, :, j * max_channel_size:(j + 1) * max_channel_size]
            batch_y_subset = batch_y[:, :, j * max_channel_size:(j + 1) * max_channel_size]

            # if 'PEMS' in cfg.model.data or 'Solar' in cfg.model.data:
            #     batch_x_mark = None
            #     batch_y_mark = None
            # else:
            # batch_x_mark = batch_x_mark.to(device)
            # batch_y_mark = batch_y_mark.to(device)

            # Decoder input
            
            dec_inp = torch.zeros_like(batch_y_subset[:, -self.kwargs.pred_len:, :], device=self.kwargs.device).float()
            dec_inp = torch.cat([batch_y_subset[:, :self.kwargs.label_len, :], dec_inp], dim=1).float()
            
            # ------ Workout this if needed -----
            # encoder - decoder
            # if cfg.model.use_amp:
            #     with torch.cuda.amp.autocast():
            #         if cfg.model.output_attention:
            #             outputs = model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]
            #         else:
            #             outputs = model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)
            # else:
            #     if cfg.model.output_attention:
            #         outputs = model(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)[0]

            #     else:
            # ------ Workout above if needed -----

            outputs = self.backbone(batch_x_subset, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.kwargs.features_mode == 'MS' else 0

            outputs = outputs[:, -self.kwargs.pred_len:, f_dim:]
            batch_y_subset = batch_y_subset[:, -self.kwargs.pred_len:, f_dim:]
            
            outputs = outputs.detach().cpu().numpy()
            batch_y_subset = batch_y_subset.detach().cpu().numpy()

            if pred is None:
                pred = outputs
                true = batch_y_subset
            else:
                pred = np.concatenate((pred, outputs), axis=2)
                true = np.concatenate((true, batch_y_subset), axis=2)

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {
            'predictions': outputs,
            'targets': batch_y_subset
        }

        
        return model_out