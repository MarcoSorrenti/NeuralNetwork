import numpy as np

class Tanh():

    def __init__(self, a=2):
        self.a = a

    def evaluate(self, x):
        return np.tanh(self.a*x/2)

    def derivative(self, x):
        return 1 - np.tanh((self.a*x)/2)**2
