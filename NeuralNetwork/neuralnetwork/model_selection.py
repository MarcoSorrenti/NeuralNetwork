import os, sys
from turtle import back, delay
import numpy as np
from neuralnetwork.model.NeuralNetwork import NeuralNetwork, build_model
from copy import deepcopy
from itertools import product
from timeit import default_timer as timer
from joblib import Parallel, delayed, parallel_backend

import warnings
warnings.filterwarnings('ignore')

class KFoldCV:
    '''KFold Cross-validation class.
    Parameters:
        model: neural network built and compiled model
        X: training set features
        y: training set target
        epochs: training epochs
        batch_size: int. batch size
        es: early stopping object
        shuffle: boolean.
    '''
    def __init__(self, model:NeuralNetwork, X, y, k_folds=4, epochs=400, batch_size=128, es=None,shuffle=False):
        self.X = X
        self.y = y

        if shuffle: self.shuffle()

        #splitting into k folds
        self.x_folds = np.array_split(X, k_folds)
        self.y_folds = np.array_split(y, k_folds)

        self.cv_valid_loss = list()
        self.cv_train_loss = list()
        self.model = model

        #cross-validation
        for i in range(k_folds):
            X_train, y_train, X_valid, y_valid = self.get_folds(i)

            #model fit for training
            try:
                with HiddenPrints():
                    history = self.model.fit(epochs=epochs,batch_size=batch_size,X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,es=es)

                train_loss = history['train_loss'][-1]
                val_loss = history['valid_loss'][-1]
                self.cv_train_loss.append(train_loss)
                self.cv_valid_loss.append(val_loss)

            except os.error:
                print("Error in training at iteration {}.".format(i))
                print(os.error)
                continue

            print("[CV {}/{}]\tSCORE --> T: {}\tV: {}".format(i+1,k_folds,train_loss,val_loss))

            #get a pre-fit copy of the default model, built and compiled
            self.model.reset_model()

        self.mean_train_loss = np.mean(self.cv_train_loss)
        self.st_dev_train = np.std(self.cv_train_loss)
        self.mean_valid_loss = np.mean(self.cv_valid_loss)
        self.st_dev_valid = np.std(self.cv_valid_loss)

        

    def get_folds(self, val_fold_id):
        '''Function for folds merging useful to build the training and validation set for each kfold iteration
        Args:
            val_fold_id: iteration. fold to exploit as validation set
        Returns:
            X_train: training set features
            y_train: training set target
            X_valid: validation set features 
            y_valid: validation set target
        '''

        X_train = np.concatenate(np.delete(self.x_folds, val_fold_id, axis=0))
        y_train = np.concatenate(np.delete(self.y_folds, val_fold_id, axis=0))
        X_valid = self.x_folds[val_fold_id]
        y_valid = self.y_folds[val_fold_id]

        assert (len(X_train == len(y_train))), "Error in folds division."
        assert (len(X_train) == len(self.X) - len(X_valid)), "Error in folds division."
        
        return X_train, y_train, X_valid, y_valid

    
    def shuffle(self):
        perm = np.random.permutation(len(self.X))
        self.X, self.y = self.X[perm], self.y[perm]




class GridSearchCVNN:
    '''Default Grid Search class for model selection (check GridSearchCVNNParallel)'''
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
                            nesterov=config['nesterov'],
                            lambd=config['lambda'])

            cv = KFoldCV(model, X=X, y=y, k_folds=k_folds, epochs=epochs, batch_size=config['batch_size'], shuffle=shuffle)

            end = timer()
            time_it = end-start

            self.grid_results.append({  'parameters':config,
                                        'mean_error_train':cv.mean_train_loss,
                                        'mean_error_valid':cv.mean_valid_loss,
                                        'st_dev_train':cv.st_dev_train,
                                        'st_dev_valid':cv.st_dev_valid,
                                        'time':time_it,
                                        })



class GridSearchCVNNParallel:
    '''Parallelization of the standard Grid Search class
    Parameters:
        param_grid: dict. parameter grid with parameters combinations. A list of values for each parameter.
    '''
    def __init__(self, params_grid:dict):
        self.param_grid = params_grid
        self.configurations = list(product(*params_grid.values()))
        self.grid_results = list()


    def fit(self, X, y, loss='mse', scoring=None, k_folds=4, epochs=100, shuffle=False, n_jobs=1):
        '''Grid search execution
        Args:
            X: training matrix
            y: training target
            loss: training loss for model compile for each parameters combination
            scoring: evaluation metric for model compile for each parameters combination
            k_folds: number of folds for the k-fold cross validation
            epochs: training epochs for the model compile
            shuffle: boolean. shuffle control for k-fold cv
            n_jobs: number of cores to exploit for the parallel grid search
        '''

        self.X = X
        self.y = y
        self.loss = loss
        self.scoring = scoring
        self.k_folds = k_folds
        self.epochs = epochs
        self.shuffle = shuffle

        print("Fitting {} folds for each of {} parameters configurations".format(k_folds, len(self.configurations)))

        results = Parallel(n_jobs=n_jobs)( delayed(self.train_config)(i) for i in range(len(self.configurations)) )

        self.grid_results = results
        

    def train_config(self, i):
        '''K-fold cross-validation for a single configuration
        Args:
            i: parameters configuration
        Returns:
            result: dict. mean loss, mean score, standard deviation for both loss and score on training and validation sets; computation time too.
        '''
        start = timer()    
        config = self.configurations[i]
        config = {key:value for key,value in zip(self.param_grid.keys(), config)}
        
        print("Configuration {}:\t{}".format(i+1, config))

        model = build_model(config)
        model.compile('sgd', 
                        loss=self.loss, 
                        metric=self.scoring, 
                        lr=config['lr'],
                        momentum=config['momentum'],
                        reg_type=config['reg_type'],
                        lr_decay=config['lr_decay'],
                        nesterov=config['nesterov'],
                        lambd=config['lambda'])

        cv = KFoldCV(model, X=self.X, y=self.y, k_folds=self.k_folds, epochs=self.epochs, batch_size=config['batch_size'], es=config['es'], shuffle=self.shuffle)

        end = timer()
        time_it = end-start

        result = {  
                'parameters':config,
                'mean_error_train':cv.mean_train_loss,
                'mean_error_valid':cv.mean_valid_loss,
                'st_dev_train':cv.st_dev_train,
                'st_dev_valid':cv.st_dev_valid,
                'time':time_it,
                }
        
        return result


class HiddenPrints:
    '''Class for hiding kfold single trainings prints'''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

