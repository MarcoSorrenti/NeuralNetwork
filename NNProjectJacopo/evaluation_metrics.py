import numpy as np

def confusion_matrix(y_true, y_preds):
    '''
    Computing confusion matrix function.

    Parameters
    ----------

    y_true : 2d matrix. Ground truth (correct) labels.
    y_pred: 2d matrix. Output labels returned by a classifier. 

    Returns
    ----------
    
    :cm: n-dimensional confusion matrix where 'n' is the number of different target labels.
    '''
    n_labels = len(np.unique(y_true))
    cm = np.zeros((n_labels,n_labels))

    for i in range(len(y_true)):
        cm[y_true[i],y_preds[i]] += 1

    return cm




def accuracy_metric(y_true, y_preds):
    
    cm = confusion_matrix(y_true, y_preds)
    
    tp = cm[0,0]
    fp = cm[1,0]
    tn = cm[1,1]
    fn = cm[0,1]
    
    accuracy = (tp + tn) / (tp+tn+fp+fn)
    return accuracy * 100.

def precision_metric(y_true, y_preds):
    
    cm = confusion_matrix(y_true, y_preds)
    
    tp = cm[0,0]
    fp = cm[1,0]
    tn = cm[1,1]
    fn = cm[0,1]
    
    precision = tp / (tp + fp)
    return precision * 100.

def recall_metric(y_true, y_preds):
    
    cm = confusion_matrix(y_true, y_preds)
    
    tp = cm[0,0]
    fp = cm[1,0]
    tn = cm[1,1]
    fn = cm[0,1]
    
    recall = tp / (tp + fn)
    return recall * 100.

def f1score_metric(y_true, y_preds):
    
    cm = confusion_matrix(y_true, y_preds)
    
    tp = cm[0,0]
    fp = cm[1,0]
    tn = cm[1,1]
    fn = cm[0,1]
    
    p = precision_metric(y_true, y_preds)
    r = recall_metric(y_true, y_preds)
    
    f1score = 2 * (p * r) / (p + r)
    return f1score

