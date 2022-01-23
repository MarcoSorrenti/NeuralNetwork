import numpy as np

class L1():
    '''Lasso regression'''
    def __init__(self, lambd):
        self.lambd = lambd

    def compute(self, w):
        return self.lambd * np.sum(np.abs(w)) #to be checked

    def derivate(self, w_old):
        return 2 * self.lambd * w_old

class L2():
    '''Ridge regression'''
    def __init__(self, lambd=0.001):
        self.lambd = lambd

    def compute(self, w):
        return self.lambd * np.sum((w)**2) #to be checked

    def derivate(self, w_old):
        return self.lambd * w_old

regularization_dict = {
    "l1": L1,
    "l2": L2
}

