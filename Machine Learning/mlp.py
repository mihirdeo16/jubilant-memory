#!/usr/bin/env python3
"""
Multi-layer Perceptron
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np


def sigmoid_fn(data:np.ndarray)-> np.ndarray:
    return 1 / (1+ np.exp(data))

def relu_fn(data:np.ndarray) -> np.ndarray:
    return np.maximum(0,data)

def softmax_fn(data:np.ndarray) -> np.ndarray:
    return np.exp(data)/np.sum(np.exp(data),axis=1,keepdims=True)

def linear_activation_fn(data:np.ndarray) -> np.ndarray:
    return data

# Extras
def leaky_relu_fn(data:np.ndarray,slop=0.1) -> np.ndarray:
    return np.where(data>=0, data, data*slop)

def tanh_fn(data:np.ndarray)-> np.ndarray:
    return np.divide((np.exp(data) - np.exp(-data)), (np.exp(data) + np.exp(-data)))

def softplus_fn(data:np.ndarray)-> np.ndarray:
    return np.log(np.exp(data)+1)

def softsign_fn(data:np.ndarray)-> np.ndarray:
    return np.divide(data,np.abs(data)+1)

def elu_fn(data:np.ndarray,alpha_:np.float16)-> np.ndarray:
    return np.where(data>=0,data,alpha_*(np.exp(-data)-1))

def selu_fn(data:np.ndarray,lambda_:np.float16,alpha_:np.float16)-> np.ndarray:
    return np.where(data>=0,data*lambda_,lambda_*(np.exp(-data)-1)*alpha_)


class LayerDense:
    def __init__(self,input_size,output_size) -> None:
        self.weights = np.random.randn(input_size,output_size) * np.sqrt(2.0 / input_size) 
        self.biases = np.random.randn(output_size)
    def forward(self,input_data) -> np.ndarray:
        output = np.dot(input_data,self.weights) + self.biases
        return output

    
class MultilayerPerceptron(LayerDense):

    def __init__(self,input_size:int,hidden_size:int,output_type:str,num_classes:int=10) -> None:
        self.hidden_layer = LayerDense(input_size,hidden_size)

        if output_type == "binary":
            output_size = 1
            self.activation_fn = sigmoid_fn
        elif output_type == "multiclass":
            output_size = num_classes  
            self.activation_fn = softmax_fn
        elif output_type == "linear":
            output_size = 1  
            self.activation_fn = linear_activation_fn
        else:
            raise ValueError("Invalid output_type. Valid options are: 'binary', 'multiclass', 'linear'")
        
        self.output_layer = LayerDense(hidden_size, output_size)
        
    def forward(self,input_data) -> np.ndarray:

        output_1 = self.hidden_layer.forward(input_data)
        output_1 = relu_fn(output_1)
        
        output_2 = self.output_layer.forward(output_1)
        
        return self.activation_fn(output_2)