import numpy as np

class RandomInit():
    def init(self, fan_in, fan_out):
        limit = 0.25            # could be a parameter?
        weights = np.random.uniform(-limit,limit,(fan_in,fan_out))
        return weights

class XavierNormal():
    def init(self, fan_in, fan_out):
        limit = np.sqrt(2 / float(fan_in + fan_out))
        weights = np.random.normal(0., limit, size=(fan_in, fan_out))
        return weights

class XavierUniform():
    def init(self, fan_in, fan_out):
        limit = np.sqrt(6 / float(fan_in + fan_out))
        weights = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
        return weights

weights_init_dict = {
    "random_init": RandomInit(),
    "xavier_normal": XavierNormal(),
    "xavier_uniform": XavierUniform()
    }