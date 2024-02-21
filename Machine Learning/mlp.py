#!/usr/bin/env python3
"""
Multi-layer Perceptron
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from turtle import forward
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

# Loss function
def categorical_cross_entropy(true_output, predicted_output):
    return -np.mean(np.sum(true_output * np.log(predicted_output), axis=1))

def binary_cross_entropy(true_output, predicted_output):
    return -np.mean(true_output * np.log(predicted_output) + (1 - true_output) * np.log(1 - predicted_output))

def mean_square_error(true_output, predicted_output):
    return np.mean(np.sum(np.square(true_output -predicted_output )))


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
        self.output_type =  output_type
        if self.output_type == "binary":
            output_size = 1
            self.activation_fn = sigmoid_fn
            self.loss_fn = binary_cross_entropy
        elif self.output_type == "multiclass":
            output_size = num_classes  
            self.activation_fn = softmax_fn
            self.loss_fn = categorical_cross_entropy
        elif self.output_type == "linear":
            output_size = 1  
            self.activation_fn = linear_activation_fn
            self.loss_fn = mean_square_error
        else:
            raise ValueError("Invalid output_type. Valid options are: 'binary', 'multiclass', 'linear'")
        
        self.output_layer = LayerDense(hidden_size, output_size)
        
    def forward(self):

        self.output_1 = self.hidden_layer.forward(self.input_data)
        self.act_output_1 = relu_fn(self.output_1)
        
        self.output_2 = self.output_layer.forward(self.act_output_1)
        self.act_output_2 = self.activation_fn(self.output_2)
            
    def back_propagation(self):

        output_weights_gradient = np.dot(self.act_output_1.T,self.loss)
        output_biases_gradient = np.sum(self.loss,axis=0)

        self.output_layer.weights -= self.learning_rate * output_weights_gradient
        self.output_layer.biases -= self.learning_rate * output_biases_gradient

        hidden_error = np.dot(self.loss, self.output_layer.weights.T) * (self.output_1 > 0)

        hidden_weights_gradient = np.dot(self.input_data.T,hidden_error)
        hidden_biases_gradient = np.sum(hidden_error,axis=0)

        self.hidden_layer.weights -= self.learning_rate * hidden_weights_gradient
        self.hidden_layer.biases -= self.learning_rate * hidden_biases_gradient

    def create_batches(self,x_train, y_train, batch_size):
        num_samples = len(x_train)
        for i in range(0, num_samples, batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            yield x_batch, y_batch

    def fit(self,x_train, y_train, epochs=10,batch_size=32,learning_rate=0.2):

        self.learning_rate = learning_rate
        
        for _ in range(epochs):

            for input_data, target in self.create_batches(x_train, y_train, batch_size):
                
                self.input_data = input_data
                self.target = target

                self.forward()

                self.loss = self.loss_fn(self.act_output_2, self.target)

                self.back_propagation()