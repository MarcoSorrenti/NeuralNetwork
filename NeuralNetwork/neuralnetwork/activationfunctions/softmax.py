import numpy as np

class Softmax():

    def evaluate(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)

    def derivative(self, x):
        return np.diagflat(self.evaluate(x)) - np.dot(self.evaluate(x), self.evaluate(x).T)