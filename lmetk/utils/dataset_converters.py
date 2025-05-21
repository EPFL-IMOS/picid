import torch
from typing import Dict, List, Tuple, Any
import numpy as np
from einops import rearrange
from abc import abstractmethod
from tqdm import tqdm


def cut_create_tensords(batch_size, *T):
    """Only acts on first dimension (B)""" 
    assert  len(np.unique(np.array([len(t) for t in T]))) == 1, "All t should have same length"
    i_max = int(len(T[0]) / batch_size) * batch_size
    T = [t[0:i_max] for t in T]
    XY_tensor = torch.utils.data.TensorDataset(*T)
    return (*T, XY_tensor)


# f is of shape samples, nodes, features, time
def fast_autoregressive(F, num_timesteps_in: int, num_timesteps_out: int, dtype):
    """ Unfolds a tensor of the shape (N_samples, ...) to be of the shape 
        (N_samples, ..., num_timesteps_in) and (N_samples, ..., num_timesteps_out)
    """

    assert torch.is_tensor(F)

    N_samples = F.shape[0]
    end_in = N_samples - (num_timesteps_in + num_timesteps_out) + 1
    start_out = num_timesteps_in

    f = F.unfold(dimension=0, size=num_timesteps_in, step=1)
    if num_timesteps_in != num_timesteps_out:
        t = F.unfold(dimension=0, size=num_timesteps_out, step=1)
    else:
        t = f

    return f[:end_in].type(dtype), t[start_out:].type(dtype)


class BatchSequencerDataset(torch.utils.data.Dataset):
    """ Naming:
        - Preloaded: original data is in memory
        - Efficient: we only window the current batch (data efficient)
        - Autoregressive: we map to ourselfes
        
        Warning: to be used with a batchsampler!      
    """
    def __init__(self, 
            Y: torch.Tensor, 
            ts_in: int,
            ts_out:int,
            X: torch.Tensor=None,
            Y_sequencer_args: Dict[str, Any] = {},
            X_sequencer_args: Dict[str, Any] = {}) -> None:
        """ Dataset for sequencer-processed datasets

        Args:
            Y (torch.Tensor): Node-local data
            X (torch.Tensor, optional): Node-independent, global data. Defaults to None.
            Y_sequencer_args (Dict[str, Any], optional): arguments passed to the sequencer. Defaults to {}.
            X_sequencer_args (Dict[str, Any], optional): arguments passed to the sequencer. Defaults to {}.
        """
        
        self.Y_sequencer = EfficientAutoregressiveSequencer(
            F=Y, ts_in=ts_in, ts_out=ts_out, **Y_sequencer_args)
        
        if X is not None:
            self.X_sequencer = GlobalFeaturesSequencer(
            F=X, ts_in=ts_in, ts_out=ts_out, **X_sequencer_args)
            assert len(self.X_sequencer) == len(self.Y_sequencer)
            assert len(Y) == len(X)
        else:
            self.X_sequencer = None
        
        self.N_samples = Y.shape[0]
        
        super().__init__()

    def __len__(self):
        return len(self.Y_sequencer)

    def __getitem__(self, batch_idx: List[int]): 
        Y_f, Y_t = self.Y_sequencer.sequences_batch(batch_idx)
        if self.X_sequencer is None:
            return Y_f, Y_t
        else:
            X_f, X_t = self.X_sequencer.sequences_batch(batch_idx) 
            return X_f, Y_f, X_t, Y_t
      
      

class IndexSequencer:
    
    def __init__(
        self, F: torch.Tensor,  
        ts_in: int, 
        ts_out: int,
        dt_ts_out: List[int] = None,
        step: int = 1) -> None:
        
        self.F = F
        assert torch.is_tensor(F)

        if dt_ts_out is not None:
            assert len(dt_ts_out) == ts_out
            dt_ts_out.sort() # inplace operation
            dt_ts_out = np.array(dt_ts_out)
            dt_max = dt_ts_out[-1]
            # we need to add 1 to dt_max because its an index,
            # while num_timesteps_out is a count
            upper_limit = max(ts_out, dt_max + 1)
        else:
            upper_limit = ts_out

        # N_samples x N_nodes x N_features
        N_samples = F.shape[0]

        # self.start_end_index_old = np.array([
        #     (i, i + (ts_in + upper_limit))
        #     for i in tqdm(range(N_samples - (ts_in + upper_limit) + 1))
        # ])
        
        i = torch.arange(0, N_samples - (ts_in + upper_limit) + 1, step=step)
        self.start_end_index = torch.stack ([i, i + (ts_in + upper_limit)], dim=1)
        
        self.N_sequences = len(self.start_end_index)
        
    def __len__(self):
        return self.N_sequences
      
      
    def sequences_batch(self, index_subset: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor]:
        sel = self.start_end_index[index_subset]
        return self._create_sequences(sel)
    
    @abstractmethod
    def _create_sequences(self, sel: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        pass
    
    

class GlobalFeaturesSequencer(IndexSequencer):
    
    def __init__(
            self, F: torch.Tensor, 
            ts_in: int, 
            ts_out: int, 
            output_shape: str = "S F T", 
            dt_ts_out: List[int] = None) -> None:
        
        self.dt_ts_out = dt_ts_out
        self.ts_in = ts_in
        
        super().__init__(F, ts_in=ts_in, ts_out=ts_out, dt_ts_out=dt_ts_out)
        self.rearrange_str = f"S T F -> {output_shape}"

    def _create_sequences(self, sel: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        ts_in, dt_ts_out = self.ts_in, self.dt_ts_out
        F = self.F
        
        # Generate observations
        f_unfolded, t_unfolded = [], []
        for i, j in sel:
            f_unfolded.append(F[i : i + ts_in, :])
            if dt_ts_out is None:
                t_unfolded.append(F[i + ts_in : j, :])
            else:
                # dt_timesteps_out is an index array
                t_unfolded.append(F[i + ts_in + dt_ts_out, :])
        
        f_unfolded = torch.stack(f_unfolded, dim=0)
        t_unfolded = torch.stack(t_unfolded, dim=0)
        # t f have now the shape S T F
        
        f_unfolded = rearrange(f_unfolded, self.rearrange_str)
        t_unfolded = rearrange(t_unfolded, self.rearrange_str)
        
        return f_unfolded, t_unfolded


class AutoregressiveSequencer(IndexSequencer):
    """A basic version of EfficientAutoregressiveSequencer without node masking
    mainly used for timeseries without node associated features. 
    
    Works for 1 to K dimensions!
    
    Test code:
    ars = AutoregressiveSequencer(torch.zeros(size=(1000,)), 3, 20, step=1)

    for i in range(len(ars)-1):
        x,y = ars.sequences_batch([i, i+1])
        print(x.shape, y.shape)
        
    """
    
    def __init__(
        self, F: torch.Tensor,
        ts_in: int, 
        ts_out: int,
        T: torch.Tensor=None,
        in_features_mask=None, 
        out_features_mask=None,
        step: int=10) -> None:
        
        super().__init__(F=F, ts_in=ts_in, ts_out=ts_out, step=step)
        self.ts_in = ts_in
        self.ts_out = ts_out
        self.in_features_mask = in_features_mask
        self.out_features_mask = out_features_mask
        self.T = F
    
    def _create_sequences(self, sel: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        # TODO: maybe we can use torch.gather somehow here
        ts_in, ts_out = self.ts_in, self.ts_out
        F, T = self.F, self.T
        
        # Generate observations
        f_unfolded, t_unfolded = [], []
        for i, j in sel:
            f_unfolded.append(F[i : i + ts_in, ...])
            t_unfolded.append(T[i + ts_in : j, ...])
        
        f_unfolded = torch.stack(f_unfolded, dim=0)
        t_unfolded = torch.stack(t_unfolded, dim=0)
        
        if self.in_features_mask is not None:
            f_unfolded = f_unfolded[..., self.in_features_mask]
            
        if self.out_features_mask is not None:
            t_unfolded = t_unfolded[..., self.out_features_mask]
        
        return f_unfolded, t_unfolded 
        
        
class EfficientAutoregressiveSequencer(IndexSequencer):
    """Data efficient autoregressive sequencer
    """
    
    def __init__(
        self, F: torch.Tensor,
        ts_in: int, 
        ts_out: int,
        in_nodes_mask=None, 
        out_nodes_mask=None,
        in_features_mask=None, 
        out_features_mask=None,
        T: torch.Tensor=None,
        output_shape: str = "S N F T",
        dt_ts_out: List[int] = None,
        step:int=1) -> None:
        """_summary_

        Args:
            F (torch.Tensor): should be in shape: Samples x Nodes x Features (feature)
            F (torch.Tensor, Optional): should be in shape: Samples x Nodes x Features (target)
            output_shape: can have any order of S N F T compatible with einops rearrange
        """
        
        super().__init__(F=F, ts_in=ts_in, ts_out=ts_out, dt_ts_out=dt_ts_out, step=step)
        
        # here we precompute all selection masks to be efficient during training...
        all_slice = slice(None, None, None)
        if in_nodes_mask is not None and in_features_mask is not None:
            # we have to do advanced mask selection to be efficient  because we cannot select
            # both dimensions in one step (see np advanced indexing chapter)
            self.in_2d_sel = torch.tensor(in_nodes_mask[..., np.newaxis] @ in_features_mask[..., np.newaxis].T)
            self.in_N_active_nodes = (in_nodes_mask == True).sum()
            self.in_N_active_features = (in_features_mask == True).sum()
        else:
            self.in_node_slice = all_slice if in_nodes_mask is None else in_nodes_mask
            self.in_feat_slice = all_slice if in_features_mask is None else in_features_mask
            self.in_2d_sel = None
            
        if out_nodes_mask is not None and out_features_mask is not None:
            self.out_2d_sel = torch.tensor(out_nodes_mask[..., np.newaxis] @ out_features_mask[..., np.newaxis].T)
            self.out_N_active_nodes = (out_nodes_mask == True).sum()
            self.out_N_active_features = (out_features_mask == True).sum()
        else:
            self.out_node_slice = all_slice if out_nodes_mask is None else out_nodes_mask
            self.out_feat_slice = all_slice if out_features_mask is None else out_features_mask
            self.out_2d_sel = None            
        
        self.ts_in = ts_in
        self.dt_ts_out = dt_ts_out
        self.T = F if T is None else T

        self.rearrange_str = f"S T N F -> {output_shape}"
        self.rearrange_str_2d = f"S T (N F) -> {output_shape}"
    

    def _create_sequences(self, sel: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        ts_in, dt_ts_out = self.ts_in, self.dt_ts_out
        F, T = self.F, self.T
        
        # Generate observations
        f_unfolded, t_unfolded = [], []
        for i, j in sel:
            f_unfolded.append(F[i : i + ts_in, :, :])
            if dt_ts_out is None:
                t_unfolded.append(T[i + ts_in : j, :, :])
            else:
                # dt_timesteps_out is an index array
                t_unfolded.append(T[i + ts_in + dt_ts_out, :, :])
        
        f_unfolded = torch.stack(f_unfolded, dim=0)
        t_unfolded = torch.stack(t_unfolded, dim=0)
        # t f have now the shape S T N F
        if self.in_2d_sel is None:
            f_unfolded = rearrange(f_unfolded[:,:,self.in_node_slice, self.in_feat_slice], self.rearrange_str)
        else:
            f_unfolded = rearrange(f_unfolded[:,:,self.in_2d_sel], 
                self.rearrange_str_2d, N=self.in_N_active_nodes, F=self.in_N_active_features)
            
        if self.out_2d_sel is None:  
            t_unfolded = rearrange(t_unfolded[:,:,self.out_node_slice, self.out_feat_slice], self.rearrange_str)
        else:
            t_unfolded = rearrange(t_unfolded[:,:,self.out_2d_sel], 
                self.rearrange_str_2d, N=self.out_N_active_nodes, F=self.out_N_active_features)

        return f_unfolded, t_unfolded 