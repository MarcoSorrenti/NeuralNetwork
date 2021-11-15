import numpy as np

def Xavier_normal(fan_in, fan_out):
    limit = np.sqrt(2 / float(fan_in + fan_out))
    weights = np.random.normal(0., limit, size=(fan_in, fan_out))
    return weights

def Xavier_uniform(fan_in, fan_out):
    limit = np.sqrt(6 / float(fan_in + fan_out))
    weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
    return weights

def He_normal(fan_in, fan_out):
    limit = np.sqrt(2/fan_in)
    weights=np.random.normal(0., limit, (fan_in,fan_out))
    return weights

def He_uniform(fan_in, fan_out):
    limit = np.sqrt(6 / float(fan_in))
    weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
    return weights


weights_initializers = {
    'xavier_normal': Xavier_normal,
    'xavier_uniform': Xavier_uniform,
    'he_normal': He_normal,
    'he_uniform': He_uniform
}

