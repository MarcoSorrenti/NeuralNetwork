from os import error
import numpy as np
from random import randrange
from tqdm import tqdm
from neuralnetwork.model.NeuralNetwork import NeuralNetwork
from copy import deepcopy

class KFoldCV:
    def __init__(self, model:NeuralNetwork, X, y, k_folds=5, epochs=400, batch_size=128):
        self.X = X
        self.y = y
        self.x_folds = np.array_split(X, k_folds)
        self.y_folds = np.array_split(y, k_folds)

        self._results = dict()
        self.model = model
        self.compiled_model = deepcopy(model)

        for i in range(k_folds):
            X_train, y_train, X_valid, y_valid = self.get_folds(i)

            try:
                history = self.model.fit(epochs=epochs,batch_size=batch_size,X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)

                self._results.update({'split_{}'.format(i+1):history})
            except error:
                print("Error in training at iteration {}.".format(i))
                print(error)
                continue

            #get a pre-fit copy of the default model, built and compiled
            self.model = self.compiled_model



    def get_folds(self, val_fold_id):
        #implement stratification for classification

        X_train = np.concatenate(np.delete(self.x_folds, val_fold_id, axis=0))
        y_train = np.concatenate(np.delete(self.y_folds, val_fold_id, axis=0))
        X_valid = self.x_folds[val_fold_id]
        y_valid = self.y_folds[val_fold_id]

        print(len(X_train), len(y_train), len(X_valid), len(y_valid))

        assert (len(X_train == len(y_train))), "Error in folds division."
        assert (len(X_train) == len(self.X) - len(X_valid)), "Error in folds division."
        
        return X_train, y_train, X_valid, y_valid




class KFoldCVRob:

    def __init__(self, folds=4):
        self.folds = folds

    def split(self, X, y):
        ds_split = list()
        X = X
        fold_size = len(X) / self.folds

        for i in range(self.folds):
            fold = []
            while len(fold) < fold_size:
                index = randrange(len(X))
                fold.append(X[index])
                np.delete(X, index)
            ds_split.append(fold)

        return ds_split






