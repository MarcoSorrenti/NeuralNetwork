import numpy as np
from copy import deepcopy

def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1.0-sigmoid(x))
    
    return 1./(1.0+np.exp(-x))

def tanh(x, derivative=False):
    t = np.tanh(x)
    if derivative:
        return 1-np.power(t,2)

    return t
    
def reLu(x, derivative=False):
    if derivative:
        return 1*(x>0)
    
    return x*(x>0)

def softmax(x, derivative=False):
    if derivative:
        pass
    
    exp_vals = np.exp(x-np.max(x, axis=1, keepdims=True))
    norm_sum = np.sum(exp_vals, axis=1, keepdims=True)
    return exp_vals/norm_sum

def sign(inputs):
    new_inputs = deepcopy(inputs)
    new_inputs[new_inputs >= 0.5] = 1
    new_inputs[new_inputs < 0.5] = 0
    
    return new_inputs.astype(int)


activation_functions_dict = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': reLu,
    'softmax': softmax
}