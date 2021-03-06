import numpy as np
from copy import deepcopy
from neuralnetwork.utils.weights_initalizer import weights_init_dict
from neuralnetwork.utils.activation import activation_function_dict


class Layer():
    '''Class representing a layer of a neural network
    Parameters:
        input_dim : layer units
        n_units : output units
        activation : type of activation function for forward pass
        weights_init : weights initialization type
    '''
    def __init__(self, input_dim, n_units, activation_function='sigmoid', weights_init='random_init'):
        self.input_dim = input_dim
        self.n_units = n_units
        self.activation_function = activation_function_dict[activation_function]
        self.weights_init = weights_init
        self.set_parameters()


    def set_parameters(self):
        '''Parameters initialization setting
        Returns:
            w : weights
            b : biases
            w_gradient : weights' gradients
            b_gradient : biases' gradients
            old_w_gradient : weights' old gradient for momentum computation
            old_b_gradient : biases' old gradient for momentum computation
        '''
        self.w = weights_init_dict[self.weights_init](self.input_dim, self.n_units)
        self.b = weights_init_dict[self.weights_init](1, self.n_units)
        self.w_gradient = np.zeros((self.input_dim,self.n_units))
        self.b_gradient = np.zeros((1,self.n_units))
        self.old_w_gradient = np.zeros((self.input_dim,self.n_units))
        self.old_b_gradient = np.zeros((1,self.n_units))


    def forward(self, input):
        '''Forward pass for feedforward propagation inside neural network
        Args:
            input : input data
        Returns:
            output : output data (dim: input_dim, n_units)
        '''
        self.input = input
        self.net = np.dot(input, self.w) + self.b
        self.output = self.activation_function.evaluate(self.net)

        return self.output

    def backward(self, error):
        '''Backward pass for gradient computing for backpropagation purposes
        Args:
            error : error at next layer
        Returns:
            error_j : error on self to propagate back to previous layer in the network
        '''
        
        #gradient from previous backpropagation iter copy and save as old gradient 
        self.old_w_gradient = deepcopy(self.w_gradient)
        self.old_b_gradient = deepcopy(self.b_gradient)

        der_net = self.activation_function.derivative(self.net)
        delta = np.multiply(error,der_net)
        self.w_gradient = np.dot(self.input.T, delta)
        self.b_gradient = np.sum(delta, axis=0, keepdims=True)
        error_j = np.dot(delta,self.w.T)

        return error_j


    
