import numpy as np
from copy import deepcopy
from neuralnetwork.activationfunctions.relu import Relu
from neuralnetwork.activationfunctions.sigmoid import Sigmoid
from neuralnetwork.weights_initalizers import *


class Layer():
    def __init__(self, input_dim, n_units, activation_function='sigmoid', weights_init='random_init'):
        self.input_dim = input_dim
        self.n_units = n_units
        self.activation_function = activation_function
        self.weights_init = weights_init
        self.w = np.zeros((self.input_dim, self.n_units))
        self.b = np.zeros((1,self.n_units))
        self.w_gradient = np.zeros((input_dim,n_units))
        self.b_gradient = np.zeros((1,n_units))

        self.weights_initializer()



    def weights_initializer(self):
        if self.weights_init == 'random':
            self.w = random_init(self.input_dim, self.n_units)
            self.b = random_init(1, self.n_units)
        elif self.weights_init == "xavier":
            self.w = Xavier_uniform(self.input_dim, self.n_units)
            self.b = Xavier_uniform(1, self.n_units)
        else:
            pass


    def forward(self, input):
        self.input = input
        self.net = np.dot(input, self.w) + self.b

        if self.activation_function == 'sigmoid':
            self.output = Sigmoid().evaluate(self.net)
        else:
            self.output = Relu().evaluate(self.net)
            
        return self.output

    def backward(self, error):
        self.old_w_gradient = deepcopy(self.w_gradient)
        self.old_b_gradient = deepcopy(self.b_gradient)

        if self.activation_function == 'sigmoid':
            der_net = Sigmoid().derivative(self.net)
        else:
            der_net = Relu().derivative(self.net)
        
        delta = np.multiply(error,der_net)

        self.w_gradient = np.dot(self.input.T, delta)
        self.b_gradient = np.sum(delta, axis=0, keepdims=True)
        error_j = np.dot(delta,self.w.T)

        return error_j





