#!/usr/bin/env python3
"""
Neural Network Variation 
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_size:int) -> None:
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_size,1)
    
    def forward(self,input):
        return self.linear(input)
    

class LogisticRegression(nn.Module):
    def __init__(self, input_size:int) -> None:
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_size,1)
    
    def forward(self,input):
        y_pred = nn.Sigmoid(self.linear(input))
        return y_pred
    
class MultiLayerPerceptronCls(nn.Module):
    def __init__(self, input_size:int,n_class:int) -> None:
        super(MultiLayerPerceptronCls,self).__init__()
        self.linear = nn.Linear(input_size,n_class)
    
    def forward(self,input):
        y_pred = nn.Softmax(self.linear(input))
        return y_pred

class DeepFeedForwardNetworkCls(nn.Module):
    def __init__(self, input_size:int,output_size:int) -> None:

        super(DeepFeedForwardNetworkCls,self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.output = nn.Linear(32,output_size)

    def forward(self,input):
        out_1 = nn.ReLU(self.fc1(input))
        out_2 = nn.ReLU(self.fc2(out_1))
        out_3 = nn.ReLU(self.fc3(out_2))
        y_pred = nn.Softmax(self.output(out_3))
        return y_pred

class DeepCNNCls(nn.Module):
    def __init__(self, input_channel:int,output_size:int) -> None:

        super(DeepCNNCls,self).__init__()

        # Input: 3x28x28 
        self.ccn_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=16,
                      kernel_size=3,stride=1,padding=2), # 16x30x30
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5,stride=1)         # 16x26x26
        ) 

        self.ccn_2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,
                      kernel_size=3,stride=1,padding=0),  # 32x24x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0) # 16x11x11
        )

        self.ccn_ffn = nn.Sequential(
            nn.Dropout2d(),
            nn.Flatten()
        )
        self.ffnetwork = nn.Sequential(
            nn.Linear(1936,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,output_size),
            nn.Softmax()
        )

    def forward(self,input):

        out_1 = self.ccn_1(input)

        out_2 = self.ccn_2(out_1)

        out_flatten = self.ccn_ffn(out_2)

        y_pred = self.ffnetwork(out_flatten)

        return y_pred



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

        self.downblock = nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(128,input_size),
            nn.ReLU(),
        )

    def forward(self,input_data):
        latent_vector = self.downblock(input_data)

        mu, std = self.latent_block(latent_vector), self.latent_block(latent_vector)

        reparameterization = mu + std*torch.rand_like(std) 

        output_data = self.decoder(reparameterization)

        return output_data, mu, std
    
