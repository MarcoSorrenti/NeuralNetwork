import numpy as np

class Activation:
    def evaluate(self, x):
        '''Evaluate function. Compute the activation function on x.
        Args:
            x: output of a layer 
        '''
        pass

    def derivative(self, x):
        '''Derivative function. Compute the derivative function on x.
        Args:
            x: output of a layer 
        '''
        pass

class Linear(Activation):
    '''Linear class. Provide the identity function and its derivative.'''
    def evaluate(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)

class Sigmoid(Activation):
    '''Sigmoid class. Provide the logistic sigmoid function and its derivative.'''
    def __init__(self, a=1):
        self.a = a
    
    def evaluate(self, x):
        return 1 / ( 1 + np.exp(-self.a * x))

    def derivative(self, x):
        x = self.evaluate(x)
        return self.a * x * (1 - x) 

class Tanh(Activation):
    '''Tanh class. Provide the hyperbolic tan function and its derivative.'''
    def __init__(self, a=2):
        self.a = a

    def evaluate(self, x):
        return np.tanh(self.a * x / 2)

    def derivative(self, x):
        return 1 - np.tanh((self.a *x ) / 2)**2


class Relu(Activation):
    '''Relu class. Provide the rectified linear unit function and its derivative.'''
    def evaluate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(int) # if x > 0 return 1 else 0

class LeakyRelu(Activation):
    '''LeakyRelu class. Provide the Leaky Relu function and its derivative.'''
    def evaluate(self, x):
        return 

    def derivative(self, x):
        return 

class Softmax(Activation):
    '''Softmax class. Provide the Softmax function and its derivative.'''
    def evaluate(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)

    def derivative(self, x):
        return np.diagflat(self.evaluate(x)) - np.dot(self.evaluate(x), self.evaluate(x).T)


activation_function_dict = {
    "linear": Linear(),
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "relu": Relu(),
    "leaky_relu":LeakyRelu(),
    "softmax": Softmax()
    }
    