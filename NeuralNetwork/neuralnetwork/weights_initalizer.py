import numpy as np

class WeightsInitalizer:
    def init(self, fan_in, fan_out):
        pass

class RandomInit(WeightsInitalizer):
    def init(self, fan_in, fan_out):
        limit = 0.25            # could be a parameter?
        weights = np.random.uniform(-limit,limit,(fan_in,fan_out))
        return weights

class XavierNormal(WeightsInitalizer):
    def init(self, fan_in, fan_out):
        limit = np.sqrt(2 / float(fan_in + fan_out))
        weights = np.random.normal(0., limit, size=(fan_in, fan_out))
        return weights

class XavierUniform(WeightsInitalizer):
    def init(self, fan_in, fan_out):
        limit = np.sqrt(6 / float(fan_in + fan_out))
        weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
        return weights

# def random_init(fan_in, fan_out):
#     limit = 0.25
#     weights = np.random.uniform(-limit,limit,(fan_in,fan_out))
#     return weights

# def Xavier_normal(fan_in, fan_out):
#     limit = np.sqrt(2 / float(fan_in + fan_out))
#     weights = np.random.normal(0., limit, size=(fan_in, fan_out))
#     return weights

# def Xavier_uniform(fan_in, fan_out):
#     limit = np.sqrt(6 / float(fan_in + fan_out))
#     weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
#     return weights