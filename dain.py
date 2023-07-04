import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class DAINLayer(nn.Module):

    def __init__(self, 
                 n_features: int,
                 mode: str | None = None, 
                 mean_lr: float = 0.00001, 
                 scale_lr: float = 0.00001,
                 gate_lr: float = 0.001, 
                 return_means: bool = False
                 ) -> None:
        
        super(DAINLayer, self).__init__()

        self.available_modes = {'avg', 'adaptive_avg', 'adaptive_scale', 'full'}
        if mode is not None and mode not in self.available_modes:
            raise ValueError(f'mode parameter not in {self.available_modes}, got {mode}.')

        if mode is None and return_means:
            raise ValueError('You cannot set return_means as True if mode is None.')

        self.n_features = n_features
        self.mode = mode
        self.mean_lr = mean_lr
        self.scale_lr = scale_lr
        self.gate_lr = gate_lr
        self.return_means = return_means

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(self.n_features, self.n_features, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(np.eye(self.n_features, self.n_features))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(self.n_features, self.n_features, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(np.eye(self.n_features, self.n_features))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(self.n_features, self.n_features)

        self.eps = 1e-8


    def simple_avg(self, x:Tensor) -> Tensor | tuple[Tensor, Tensor]:
        # expected shape (batch, ts_length, n_features)
        avg = torch.mean(x, dim=1)  
        avg = avg.unsqueeze(1)  # unsqueeze for the computation
        x = x - avg
        if self.return_means:
            return x, avg.squeeze(1)
        return x


    def adaptive_avg(self, x:Tensor) -> Tensor | tuple[Tensor, Tensor]:
        # expected shape (batch, ts_length, n_features)
        avg = torch.mean(x, dim=1) 
        adaptive_avg = self.mean_layer(avg)     # returns (batch, n_features)
        adaptive_avg = adaptive_avg.unsqueeze(1)
        x = x - adaptive_avg
        if self.return_means:
            return x, avg       # I'm not sure if we want to return avg or adaptive_avg
        return x
    
    
    def adaptive_std(self, x:Tensor) -> Tensor:
        # expected shape (batch, ts_length, n_features)
        std = torch.std(x, dim=1)
        adaptive_std = self.scaling_layer(std)  # returns (batch, n_features)
        #adaptive_std[adaptive_std <= self.eps] = 1  # remove d because I don't get its purpose
        adaptive_std = adaptive_std.unsqueeze(1)
        x = x / (adaptive_std + self.eps)
        return x
    

    def gating(self, x:Tensor) -> Tensor:
        # expected shape (batch, ts_length, n_features)
        avg = torch.mean(x, dim=1)
        gate = F.sigmoid(self.gating_layer(avg))    # returns (batch, n_features)
        gate = gate.unsqueeze(1)
        x = x * gate
        return x


    def forward(self, x:Tensor) -> Tensor | tuple[Tensor, Tensor]:
        # Expecting  (batch_size, ts_length, n_features)
        # steps: adaptive avg, adaptive std, gating

        if self.mode == None:
            # do nothing
            return x
        
        if self.mode == 'avg':
            return self.simple_avg(x)           # simple average normalization
        
        if self.mode == 'adaptive_avg':
            return self.adaptive_avg(x)         # step 1
        
        if self.mode == 'adaptive_scale':
            if self.return_means:
                x, mu = self.adaptive_avg(x)    # step 1
                x = self.adaptive_std(x)        # step 2
                return x, mu
            else:
                x = self.adaptive_avg(x)        #type:ignore # step 1    
                x = self.adaptive_std(x)        # step 2
                return x
        
        if self.mode == 'full':
            if self.return_means:
                x, mu = self.adaptive_avg(x)    # step 1
                x = self.adaptive_std(x)        # step 2
                x = self.gating(x)              # step 3
                return x, mu
            else:
                x = self.adaptive_avg(x)        #type:ignore # step 1 
                x = self.adaptive_std(x)        # step 2
                x = self.gating(x)              # step 3
                return x
        
        raise ValueError(f'{self.mode} not in {self.available_modes}')