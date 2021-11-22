import numpy as np

class Relu():

    def evaluate(self, x):
        return np.maximum(0,x)

    def derivative(self, x):
        return (x > 0).astype(int) # if x > 0 return 1 else 0
