import numpy as np

def RandomInit(fan_in, fan_out):
    limit = 0.25
    weights = np.random.uniform(-limit,limit,(fan_in,fan_out))
    return weights

def XavierNormal(fan_in, fan_out):
    limit = np.sqrt(2 / float(fan_in + fan_out))
    weights = np.random.normal(0., limit, size=(fan_in, fan_out))
    return weights

def XavierUniform(fan_in, fan_out):
    limit = np.sqrt(6 / float(fan_in + fan_out))
    weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
    return weights

def HeNormal(fan_in, fan_out):
    limit = np.sqrt(2 / float(fan_in))
    weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
    return weights

def HeUniform(fan_in, fan_out):
    limit = np.sqrt(6 / float(fan_in))
    weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
    return weights


weights_init_dict = {
    "random_init": RandomInit,
    "xavier_normal": XavierNormal,
    "xavier_uniform": XavierUniform,
    "he_normal":HeNormal,
    "he_uniform":HeUniform
    }