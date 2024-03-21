#!/usr/bin/env python3
"""
Neural Network Variation 
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


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
        self.sigmoid = nn.Sigmoid()
    def forward(self,input):
        output = self.linear(input)
        y_pred = self.sigmoid(output)
        return y_pred
    
class MultiLayerPerceptronCls(nn.Module):
    def __init__(self, input_size:int,n_class:int) -> None:
        super(MultiLayerPerceptronCls,self).__init__()
        self.linear = nn.Linear(input_size,n_class)
        self.softmax = nn.Softmax()
    
    def forward(self,input):
        y_pred = self.softmax(self.linear(input))
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




