import numpy as np

class Activation:
    def evaluate(self, x):
        pass

    def derivative(self, x):
        pass

class Linear(Activation):
    def evaluate(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)

class Sigmoid(Activation):
    def __init__(self, a=1):
        self.a = a
    
    def evaluate(self, x):
        return 1 / ( 1 + np.exp(-self.a * x))

    def derivative(self, x):
        x = self.evaluate(x)
        return self.a * x * (1 - x) 

class Tanh(Activation):
    def __init__(self, a=2):
        self.a = a

    def evaluate(self, x):
        return np.tanh(self.a * x / 2)

    def derivative(self, x):  
        return 1 - np.tanh((self.a *x ) / 2)**2


class Relu(Activation):
    def evaluate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(int) # if x > 0 return 1 else 0

class Softmax(Activation):
    def evaluate(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)

    def derivative(self, x):
        return np.diagflat(self.evaluate(x)) - np.dot(self.evaluate(x), self.evaluate(x).T)

