import numpy as np
from copy import deepcopy
from neuralnetwork.weights_initalizer import weights_init_dict
from neuralnetwork.activation import activation_function_dict


class Layer():
    def __init__(self, input_dim, n_units, activation_function='sigmoid', weights_init='random_init'):
        self.input_dim = input_dim
        self.n_units = n_units
        self.activation_function = activation_function_dict[activation_function]
        self.weights_init = weights_init_dict[weights_init]
        self.weights_initializer()
        self.w_gradient = np.zeros((input_dim,n_units))
        self.b_gradient = np.zeros((1,n_units))

    def weights_initializer(self):
        self.w = self.weights_init.init(self.input_dim, self.n_units)
        self.b = self.weights_init.init(1, self.n_units)
        
    def forward(self, input):
        self.input = input
        self.net = np.dot(input, self.w) + self.b

        self.output = self.activation_function.evaluate(self.net)

        return self.output

    def backward(self, error):
        self.old_w_gradient = deepcopy(self.w_gradient)
        self.old_b_gradient = deepcopy(self.b_gradient)

        der_net = self.activation_function.derivative(self.net)

        delta = np.multiply(error,der_net)

        self.w_gradient = np.dot(self.input.T, delta)
        self.b_gradient = np.sum(delta, axis=0, keepdims=True)
        error_j = np.dot(delta,self.w.T)

        return error_j





