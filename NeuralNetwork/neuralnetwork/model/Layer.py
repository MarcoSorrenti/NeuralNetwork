import numpy as np
from neuralnetwork.activationfunctions.sigmoid import Sigmoid

class Layer():
    def __init__(self, input_dim, n_units, activation_function='sigmoid', weights_init='random_init'):
        self.input_dim = input_dim
        self.n_units = n_units
        self.activation_function = activation_function
        self.weights_init = weights_init
        self.w = np.zeros((self.input_dim, self.n_units))
        self.b = np.zeros((1,self.n_units))

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
        self.output = Sigmoid().evaluate(self.net)


    def backward(self, error):
        pass






    



    