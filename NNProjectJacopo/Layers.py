import numpy as np
from copy import deepcopy
from weights_initialization import weights_initializers
from activation_functions import activation_functions_dict

class DLayer(object):
    def __init__(self, input_shape, output_shape, activation='sigmoid', weights_init='xavier_normal'):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.initializer = weights_init
        self.activation = activation
        
        self.weights = weights_initializers[weights_init](input_shape, output_shape)
        ##including bias into weights matrix??????
        self.biases = weights_initializers[weights_init](1, output_shape)
        
        self.weight_gradients = np.zeros((input_shape,output_shape))
        self.bias_gradients = np.zeros((1,output_shape))
        self.old_weight_gradients = np.zeros((input_shape,output_shape))
        self.old_bias_gradients = np.zeros((1,output_shape))
        
        
    def forward(self, inputs):
        self.inputs = inputs
        self.nets = np.dot(inputs,self.weights) + self.biases
        self.outputs = activation_functions_dict[self.activation](self.nets)
        
        return self.outputs
    
    
    def backward(self, error):
        
        #saving memory of the past gradients to exploit it in the backpropagation with momentum
        self.old_weight_gradients = deepcopy(self.weight_gradients)
        self.old_bias_gradients = deepcopy(self.bias_gradients)
        
        #in case the loss is LMS -->
        f_prime = activation_functions_dict[self.activation](self.nets, derivative=True) #derivative of activation function
        delta = np.multiply(error, f_prime)
        self.weight_gradients = np.dot(self.inputs.T, -delta)
        self.bias_gradients = np.sum(-delta, axis=0, keepdims=True)
        
        output_error = np.dot(delta,self.weights.T)
        
        return output_error
