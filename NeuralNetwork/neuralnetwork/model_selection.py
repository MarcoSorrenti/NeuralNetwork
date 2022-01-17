import os, sys
import numpy as np
from tqdm import tqdm
from neuralnetwork.model.NeuralNetwork import NeuralNetwork, build_model
from copy import deepcopy
from itertools import product
from timeit import default_timer as timer

class KFoldCV:
    def __init__(self, model:NeuralNetwork, X, y, k_folds=5, epochs=400, batch_size=128, shuffle=False):
        self.X = X
        self.y = y

        if shuffle: self.shuffle()

        self.x_folds = np.array_split(X, k_folds)
        self.y_folds = np.array_split(y, k_folds)

        self.cv_loss = list()
        self.model = model

        for i in range(k_folds):
            X_train, y_train, X_valid, y_valid = self.get_folds(i)

            try:
                with HiddenPrints():
                    history = self.model.fit(epochs=epochs,batch_size=batch_size,X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)

                loss = history['valid_loss'][-1]
                self.cv_loss.append(loss)

            except os.error:
                print("Error in training at iteration {}.".format(i))
                print(os.error)
                continue

            print("[CV {}/{}]\tSCORE: {}".format(i+1,k_folds,loss))

            #get a pre-fit copy of the default model, built and compiled
            self.model.reset_model()


        self.mean_valid_loss = np.mean(self.cv_loss)
        self.st_dev = np.std(self.cv_loss)

        

    def get_folds(self, val_fold_id):
        #implement stratification for classification

        X_train = np.concatenate(np.delete(self.x_folds, val_fold_id, axis=0))
        y_train = np.concatenate(np.delete(self.y_folds, val_fold_id, axis=0))
        X_valid = self.x_folds[val_fold_id]
        y_valid = self.y_folds[val_fold_id]

        assert (len(X_train == len(y_train))), "Error in folds division."
        assert (len(X_train) == len(self.X) - len(X_valid)), "Error in folds division."
        
        return X_train, y_train, X_valid, y_valid

    
    def shuffle(self):
        self.X = np.random.permutation(self.X)
        self.y = np.random.permutation(self.y)




class GridSearchCVNN:
    def __init__(self, params_grid:dict):
        self.param_grid = params_grid
        self.configurations = list(product(*params_grid.values()))
        self.grid_results = list()


    def fit(self, X, y, loss='mse', scoring=None, k_folds=4, epochs=100, shuffle=False):

        print("Fitting {} folds for each of {} parameters configurations".format(k_folds, len(self.configurations)))

        
        for i,config in enumerate(self.configurations):


            start = timer()    
            config = {key:value for key,value in zip(self.param_grid.keys(), config)}
            
            print("Configuration {}:\t{}".format(i+1, config))

            model = build_model(config)
            model.compile('sgd', 
                            loss=loss, 
                            metric=scoring, 
                            lr=config['lr'],
                            momentum=config['momentum'],
                            reg_type=config['reg_type'],
                            lr_decay=config['lr_decay'],
                            nesterov=config['nesterov'])

            cv = KFoldCV(model, X=X, y=y, k_folds=k_folds, epochs=epochs, batch_size=config['batch_size'], shuffle=shuffle)

            end = timer()
            time_it = end-start

            self.grid_results.append({  'parameters':config,
                                        'mean_valid_error':cv.mean_valid_loss,
                                        'st_dev_valid':cv.st_dev,
                                        'time':time_it,
                                        })


            

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
