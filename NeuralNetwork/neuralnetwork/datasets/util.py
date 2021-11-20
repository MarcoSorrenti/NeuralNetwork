import pandas as pd
import numpy as np



def import_dataset(problem):
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

    X_train, y_train = split_data(monk_train)
    X_test, y_test = split_data(monk_test)

    return X_train, X_test, y_train, y_test

def split_data(dataset):
    dataset = dataset.iloc[:,:-1]
    #monk_train = monk_train.sample(frac=1, random_state=42)

    X_to_encode = dataset.iloc[:,1:]
    #one hot encoding
    X = pd.get_dummies(X_to_encode, columns=[col for col in X_to_encode.columns]) # encoding
    y = dataset.iloc[:,0]
    X = np.array(X)
    y = np.array(y).reshape(-1,1)

    return X, y 

    