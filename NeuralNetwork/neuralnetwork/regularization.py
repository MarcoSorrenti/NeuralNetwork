import numpy as np

class L1():        
    def compute(self, lambd, w):
        return lambd * np.sum(np.abs(w)) #to be checked

    def derivate(self, lambd, w_old):
        return 2 * lambd * w_old

class L2():
    def compute(self, lambd, w):
        return lambd * np.sum((w)**2) #to be checked

    def derivate(self, lambd, w_old):
        return 2 * lambd * w_old

regularization_dict = {
    "l1": L1(),
    "l2": L2()
}

