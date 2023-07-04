import torch
from torch import Tensor
import torch.nn as nn



class GASCell(nn.Module):
    
    def __init__(self, 
                 eta_mu: float,
                 eta_var: float
                 ) -> None:
        
        super(GASCell, self).__init__()
        
        self.mu = None
        self.var = None
        self.eta_mu = eta_mu
        self.eta_var = eta_var
    
    def set_mu_var(self, mu:Tensor, var:Tensor) -> None:
        self.mu = mu
        self.var = var

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:

        if self.mu is None or self.var is None:
            raise ValueError("You must set means and variances first! Use set_mu_var method.")
        
        # expected tensor of shape (n_features)
        if x.dim() != 1:
            raise ValueError(f"Wrong number of dimensions. Expected 1, got {x.dim()}.")
        
        # update mu and var
        self.mu += self.eta_mu * (x - self.mu)
        self.var = self.var * (1 - self.eta_var) + self.eta_var * (x - self.mu)**2
        
        # normalize input
        norm_x = (x - self.mu) / torch.sqrt(self.var)

        return norm_x, self.mu, self.var



class GASLayer(nn.Module):
    
    def __init__(self, 
                 eta_mu: float, 
                 eta_var: float
                 ) -> None:
        
        super(GASLayer, self).__init__()
        
        self.gas_cell = GASCell(eta_mu, eta_var)
    
    
    def init_mu_var(self, mu:Tensor, var:Tensor) -> None:
        self.gas_cell.set_mu_var(mu, var)


    def forward(self, x:Tensor) -> tuple[Tensor, Tensor]:
        input_dim = x.dim()
        if not (input_dim == 2 or input_dim== 3):
            raise ValueError(f"Wrong number of dimensions. Expected 2 or 3, got {input_dim}.")
        
        # assumed tensor of shape (ts_length, n_features) or (1, ts_length, n_features)
        if input_dim == 3:
            if x.shape[0] != 1:
                raise ValueError("This method supports only online learning.")
            x = x.squeeze(0)
        
        # initialize mu and vars of the gas cell as first els of the time series
        mean_x = torch.mean(x, dim=0)
        std_x = torch.std(x, dim=0)
        self.init_mu_var(mean_x, std_x)

        # initialize results
        norm_x = torch.empty_like(x)
        mus = torch.empty_like(x)
        vars = torch.empty_like(x)
        
        for i in range(x.shape[0]):
            norm_x[i], mus[i], vars[i] = self.gas_cell(x[i])
        
        # return tensor of the same shape as original input
        if input_dim == 3:
            norm_x = norm_x.unsqueeze(0)
            mus = mus.unsqueeze(0)
            vars = vars.unsqueeze(0)
        
        # combine additional information into one single tensor of shape (1, ts_length, 2*n_features)
        additional_info = torch.cat((mus, vars), dim=-1)

        return norm_x, additional_info



class GASModel(nn.Module):

    def __init__(self, 
                 ts_encoder: nn.Module, # model to embed time series data
                 eta_mu: float, 
                 eta_var: float,
                 output_model: nn.Module, # downstream model
                 ) -> None:
        
        super(GASModel, self).__init__()

        self.ts_encoder = ts_encoder
        self.gas = GASLayer(eta_mu, eta_var)
        self.output_model = output_model
    

    def forward(self, x:Tensor) -> Tensor:

        # assuming shape (batch, ts_length, n_features)
        if x.dim() != 3:
            raise ValueError(f"Wrong number of dimensions. Expected 3, got {x.dim()}.")
        
        # only online mode supported 'til now
        if x.shape[0] != 1:
            raise ValueError("This method supports only online learning.")
        
        x, add_info = self.gas(x)    # tensors shape (1, ts_length, n_features) and (1, ts_length, 2*n_features)

        ####################### possibly a lot of new features!!!
        add_info = add_info.reshape(x.shape[0], -1)   # this becomes (batch, ts_length * 2* n_features)
        #######################

        # process the normalized timeseries
        x = self.ts_encoder(x)      # shape (batch, n_encoded_features)
        # check that the output_model has the correct input shape
        #output_in_dim = self.output_model.input_shape
        #log_sentence = f'Input n_feat of the output model must be encoded_dim + (ts_length * n_features), got {output_in_dim}.'
        #assert output_in_dim == (x.shape[0], x.shape[1] + mus.shape[1]), log_sentence
        
        # concatenate normalized x and the means
        x = torch.cat((x, add_info), dim=1)

        return self.output_model(x)
