import numpy as np
from random import randrange


class KFoldCV:

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

# class GridSearch():
