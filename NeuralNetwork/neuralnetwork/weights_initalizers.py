import numpy as np

def random_init(fan_in, fan_out):
    limit = 0.25
    weights = np.random.uniform(-limit,limit,(fan_in,fan_out))
    return weights

def Xavier_normal(fan_in, fan_out):
    limit = np.sqrt(2 / float(fan_in + fan_out))
    weights = np.random.normal(0., limit, size=(fan_in, fan_out))
    return weights

def Xavier_uniform(fan_in, fan_out):
    limit = np.sqrt(6 / float(fan_in + fan_out))
    weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
    return weights