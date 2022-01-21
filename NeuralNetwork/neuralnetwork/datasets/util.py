from email import header
from email.quoprimime import header_decode
import pandas as pd
import numpy as np


def load_monk(problem):
    '''Loading monk train and test data from remote URLS
    Args: 
        problem : monk problem requested
    Returns:
        X_train, X_test : feature matrices for train and test set
        y_train, y_test : target matrices for train and test set
    '''
    monk_train = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-{}.train".format(problem),
        header=None,
        sep=' ',
        skipinitialspace=True
        )
    
    monk_test = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-{}.test".format(problem),
        header=None,
        sep=' ',
        skipinitialspace=True
        )

    X_train, y_train = split_monk_data(monk_train)
    X_test, y_test = split_monk_data(monk_test)

    return X_train, X_test, y_train, y_test


def split_monk_data(dataset):
    '''Splitting function for monk data
    Args: 
        dataset : original monk dataset
    Returns:
        X, y : Feature matrix and target column
    '''
    #removing useless columns
    dataset = dataset.iloc[:,:-1]
    X_to_encode = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]
    #one hot encoding
    X = pd.get_dummies(X_to_encode, columns=[col for col in X_to_encode.columns])
    X = np.array(X)
    y = np.array(y).reshape(-1,1)

    return X, y 



def load_cup():
    '''Loading cup data from local folder
    Returns:
        X, y : Feature matrix and target column
    '''
    cup_folder = "NeuralNetwork/neuralnetwork/datasets/"
    train_file = "ML-CUP21-TR.csv"
    test_file = "ML-CUP21-TS.csv"
    cup_train = pd.read_csv(cup_folder + train_file, skiprows=7, header=None, index_col=0)
    cup_test = pd.read_csv(cup_folder + test_file, skiprows=7, header=None, index_col=0)

    X_train = np.array(cup_train.iloc[:,:10])
    y_train = np.array(cup_train.iloc[:,-2:])
    X_test = np.array(cup_test.iloc[:,:10])
    y_test = np.array(cup_test.iloc[:,-2:])

    return X_train, X_test, y_train, y_test


def train_test_split(X, y, test_size=0.15):
    '''Splitting function for dataset decomposition into train and test(validation) set
    Args: 
        X : feature matrix
        y : target array or matrix
        test_size : percentage of the original dataset assigned to the test set 
    Returns:
        X, y : Feature matrix and target column
    '''
    index_split = int(np.floor(len(X)*(1-test_size)))
    X_train, X_test = X[:index_split],X[index_split:]
    y_train, y_test = y[:index_split],y[index_split:]

    return X_train, X_test, y_train, y_test