import numpy as np
from copy import deepcopy
from neuralnetwork.activationfunctions.relu import Relu
from neuralnetwork.activationfunctions.sigmoid import Sigmoid


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


    def Xavier_normal(self, fan_in, fan_out):
        limit = np.sqrt(2 / float(fan_in + fan_out))
        weights = np.random.normal(0., limit, size=(fan_in, fan_out))
        return weights

    def Xavier_uniform(self, fan_in, fan_out):
        limit = np.sqrt(6 / float(fan_in + fan_out))
        weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
        return weights

    def weights_initializer(self):
        if (self.weights_init == "random_init"):
            self.w = self.Xavier_uniform(self.input_dim, self.n_units)
            self.b = self.Xavier_uniform(1, self.n_units)
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
            delta = error * Sigmoid().derivative(self.net)
        else:
            delta = error * Relu().derivative(self.net)

        self.w_gradient = np.dot(self.input.T, delta)
        self.b_gradient = np.sum(delta, axis=0, keepdims=True)
        error_j = np.dot(delta,self.w.T)

        return error_j





