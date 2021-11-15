import numpy as np

def BinaryCrossEntropy(y_preds, y_true):
    eps = 1e-07
    max_p = 1 - eps

    new_preds = np.clip(y_preds, a_min=eps, a_max=max_p)
    #if len(y_true.shape) == 1:
    l = len(new_preds)
    
    loss = -((y_true*np.log(new_preds)) + ((1-y_true)*np.log(1-new_preds)))
    loss = np.sum(loss)/l
    return loss 

    
def MSE(y_preds, y_true):    
    loss = np.sum(np.square(np.subtract(y_true,y_preds)))/ (2*len(y_true))
    return loss


losses_dict = {
    'binary_crossentropy': BinaryCrossEntropy,
    'mse': MSE
}
