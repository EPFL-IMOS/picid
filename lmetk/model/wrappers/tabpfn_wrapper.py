"""Wrapper for the TimePFN model."""
import torch
from .base import AbstractWrapper

from tabpfn import TabPFNRegressor

class TabPFNWrapper(AbstractWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """
    def __init__(self, **kwargs):
        backbone=TabPFNRegressor(
            ignore_pretraining_limits=kwargs['ignore_pretraining_limits'], 
            random_state=42, 
            device='cuda' if kwargs['device']=='gpu' else 'cpu',
            )
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0

        #self.train_dataloader = kwargs['train_dataloader']
        #self.val_dataloader = kwargs['val_dataloader']
        
        self.aux_context = None
        self.y_aux = None
    
    def fit(self, x, y):
        self.backbone.fit(x.cpu(), y.cpu())
    
    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------self.backbone.fit(batch_x, batch_y)
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        context_fit = batch['context']['x'].squeeze(0)
        target_fit = batch['target']['x'].squeeze(0)

        context = batch['context']['y'].squeeze(0)
        target = batch['target']['y'].squeeze(0)
        
        # Extent the context with auxiliary features to give sence of seasonality
        self.fit(context_fit, target_fit)

        # Extent the context with auxiliary features to give sence of seasonality
        outputs = self.backbone.predict(context.cpu())
            
        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {
            'predictions': torch.Tensor(outputs).reshape(target.shape),
            'targets': target
        }
        
        return model_out


import numpy as np

def generate_and_fit_polynomial(
    total_points=1580,
    desired_fit_degree=4,
    periodicity=24,
    noise_std=0.05,
    seed=42,
    num_context=20,
    skip = 700,
):
    np.random.seed(seed)

    def ground_truth_polynomial(x_scaled):
        # Higher-degree nonlinear structure
        p1 = 0.3 * x_scaled**4
        p2 = -0.2 * x_scaled**5
        p3 = 0.1 * x_scaled**6
        periodic = 0.3 * np.sin(2 * np.pi * x_scaled * periodicity)

        # Combine components
        values = p1 + p2 + p3 + periodic
        values -= values.min()
        values /= values.max()
        return values

    # --- FIX 1: Scale x to [-1, 1] ---
    x_raw = np.linspace(500, 2000, total_points)
    x_vals = np.linspace(-1, 1, total_points)  # fit over [-1, 1] for numerical stability

    # --- FIX 2: Generate target y ---
    y_clean = ground_truth_polynomial(x_vals)
    y_vals = y_clean + np.random.normal(0, noise_std, size=total_points)

    coeffs_and_errors = []
    y_true_vals = []
    # --- Fit and evaluate ---
    for i in range(skip, skip + num_context, 1):
        x_obs = x_vals[:i]
        y_obs = y_vals[:i]

        coeffs = np.polyfit(x_obs, y_obs, desired_fit_degree)
        poly = np.poly1d(coeffs)

        x_next = x_vals[i]
        y_pred = poly(x_next)
        y_true = y_vals[i]

        error = y_pred - y_true
        coeffs_and_errors.append(np.concatenate([coeffs, [error]]))
        y_true_vals.append(y_true)

    return x_raw, y_vals, np.array(coeffs_and_errors), np.array(y_true_vals)   



# This forward is very slow, aboyt 1.5 hours for railway dataset
# context_fit = batch['context']['x'].squeeze(0)
# target_fit = batch['target']['x'].squeeze(0)

# context = batch['context']['y'].squeeze(0)
# target = batch['target']['y']
# outputs = []
# for idx in range(target.shape[1]):
#     # Get the right context and target for the current index
    
#     self.fit(context_fit[idx::24,:,], target_fit[idx::24,:,])
#     # if self.aux_context==None:
#     #     desired_fit_degree = 4
#     #     _, _, context_aux, y_aux = generate_and_fit_polynomial(desired_fit_degree=desired_fit_degree)

#     #     # Generate zeros
#     #     context_aux = np.concatenate([np.zeros((context_aux.shape[0], context_fit.shape[1] - desired_fit_degree - 2)), context_aux], axis=1)
#     #     self.aux_context = torch.Tensor(context_aux).to(batch['context']['x'].device)
#     #     self.y_aux = torch.Tensor(y_aux.reshape(-1,1)).to(batch['context']['x'].device)

    
#     # context_fit = torch.cat([self.aux_context, context_fit], dim=0)
#     # target_fit = torch.cat([self.y_aux, target_fit], dim=0)

    

    
#     output = self.backbone.predict(context[idx].unsqueeze(0).cpu())
#     outputs.append(output.reshape(-1, 1))

# outputs = np.concatenate(outputs, axis=0)