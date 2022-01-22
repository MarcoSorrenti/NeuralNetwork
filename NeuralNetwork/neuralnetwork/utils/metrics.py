import numpy as np

def mse_loss(y_true, y_preds):
    '''MSE loss function. Mean squared error.
        Args:
            y_true
            y_preds
        Returns:
            mse: loss
    '''
    return np.mean(np.square(y_true - y_preds))

def mee_loss(y_true, y_preds):
    '''MEE loss function. Mean Euclidean error.
        Args:
            y_true
            y_preds
        Returns:
            mee: loss
    '''
    return np.mean(np.sum(np.square(y_true - y_preds), axis=1))

def accuracy_bin(y_true, y_preds):
    '''Accuracy bin function.
        Args:
            y_true
            y_preds
        Returns:
            accuracy
    '''
    y_preds = (y_preds > 0.5).astype(int)
    return np.round(np.equal(y_true, y_preds).mean(),8) *100


evaluation_metrics = {
    'mse':mse_loss,
    'mee':mee_loss,
    'accuracy':accuracy_bin,
}
