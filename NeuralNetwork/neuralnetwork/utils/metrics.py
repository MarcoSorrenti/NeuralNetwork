import numpy as np

def mse_loss(y_true, y_preds):
    return np.mean(np.square(y_true - y_preds))

def accuracy_bin(y_true, y_preds):

    y_preds = (y_preds > 0.5).astype(int)
    return np.round(np.equal(y_true, y_preds).mean(),8) *100


evaluation_metrics = {
    'mse':mse_loss,
    'accuracy':accuracy_bin
}
