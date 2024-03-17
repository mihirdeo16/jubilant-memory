
"""

This module contains the implementation of the AutoEncoder, SparseAutoEncoder and VariationalAutoEncoder.

"""

import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self,input_size) -> None:
        super(AutoEncoder,self).__init__()

        self.downblock = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
        )

        self.upblock = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,input_size),
            nn.ReLU(),
        )

    def forward(self,input_data) -> torch.Tensor:
        latent_vector = self.downblock(input_data)
        output_data = self.downblock(latent_vector)

        return output_data


class SparseAutoEncoder(nn.Module):
    def __init__(self,input_size) -> None:

        super(SparseAutoEncoder,self).__init__()
        self.upblock = nn.Sequential(
            nn.Linear(input_size,input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2,input_size*3),
            nn.ReLU(),
        )

        self.downblock = nn.Sequential(
            nn.Linear(input_size*3,input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2,input_size),
            nn.ReLU(),
        )

    def forward(self,input_data) -> torch.Tensor:
        sparse_vector = self.upblock(input_data)
        output_data = self.downblock(sparse_vector)
        return output_data
    

class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_size) -> None:
        
        super(VariationalAutoEncoder,self).__init__()
        
        self.downblock = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
        )

        self.latent_block = nn.Sequential(
            nn.Linear(64,20),
            nn.ReLU()
        )

        self.upblock = nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(128,input_size),
            nn.ReLU(),
        )

    def forward(self,input_data):
        latent_vector = self.downblock(input_data)

        mu, std = self.latent_block(latent_vector), self.latent_block(latent_vector)

        reparameterization = mu + std*torch.rand_like(std) 

        output_data = self.upblock(reparameterization)

        return output_data, mu, std
    
