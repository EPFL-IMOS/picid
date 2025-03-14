

from einops import rearrange
import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class GenericTabularizer(ABC):
    
    def __init__(self):
        self.tabular_data = None
    
    
    def map_batch(self, batch):
        x_c, y_c, x_t, y_t = batch
        return x_c, y_c, x_t, y_t


    def tabularize(self, 
                   dl: DataLoader, 
                   lag: int = 0,
                   map_batch_f: callable = None):
        
        X, y = [], []
        
        for batch in iter(dl):
            x_c, y_c, x_t, y_t = self.map_batch(batch)  if map_batch_f is None else map_batch_f(batch)
            x_b, y_b = self.tabularize_batch(x_c, y_c, x_t, y_t)
            X.append(x_b)
            y.append(y_b)
            
        tabular_data = {
            "X": torch.concat(X).numpy(),
            "y": torch.concat(y).numpy()
        }
        return tabular_data
        
        
    @abstractmethod    
    def tabularize_batch(self, x_c, y_c, x_t, y_t, lag=0):
            pass
        
        
    def get_context_for(self, y_test: torch.Tensor, lag: int = 0):
        return self.tabular_data["X"], self.tabular_data["y"]
    
    
class HistoryAndContextTabularizer(GenericTabularizer):

    def __init__(self):
        super().__init__()
        
    def tabularize_batch(self, x_c, y_c, x_t, y_t, lag=0):
        y_c = rearrange(y_c, 'b t f -> b (t f)')
        x_b = torch.concat([x_t[:, lag, :], y_c], dim=1)
        y_b = y_t[:,lag,:]

        return x_b, y_b
    
    
    (timesteps x N_y_input x K_support_samples)