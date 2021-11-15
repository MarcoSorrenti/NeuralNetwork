import numpy as np

#regularization
def L2_regularization(weights, lambd):
    penalty = -2*lambd*weights
    
    return penalty



#lr decay
def lr_decay(epoch, lr, factor):
    if factor is not None and (factor and epoch != 0):
        if epoch%100 == 0:
            lr *= factor

    return lr