import os
import sys

sys.path.insert(0, os.path.abspath('neuralnetwork'))

import pickle
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_cup, train_test_split
from neuralnetwork.model_selection import GridSearchCVNNParallel

X_train, X_test_blind, y_train, y_test_blind = load_cup()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

n_features = X_train.shape[1]
batch = len(X_train)

run_grid = False

if not os.path.isfile('NeuralNetwork/cup/best_config1.pickle') or run_grid:

    es = EarlyStopping(monitor='valid_loss',patience=250,min_delta=1e-23)

    params_config = {
                'n_features': [n_features],
                'n_hidden_layers':[1, 2],
                'n_units':[60, 80, 100],
                'batch_size':[128, int(batch/3), int(batch/2)],
                'out_units':[2],
                'hidden_act':['tanh'],
                'out_act':['linear'],
                'weights_init':['he_uniform'],
                'lr':[0.001, 0.005, 0.007, 0.009, 0.01],
                'momentum':[0.5, 0.6, 0.7, 0.8, 0.9],
                'reg_type': ['l2'],
                'lambda':[0.0001, 0.0003, 0.0005, 0.001],
                'lr_decay':[False],
                'nesterov':[False,True],
                'es':[es]
                }

    if __name__ == '__main__':

        gs = GridSearchCVNNParallel(params_config)
        gs.fit(X_train,y_train,loss='mee',epochs=1000,shuffle=True,n_jobs=8)
        gs_results = sorted(gs.grid_results, key = lambda i: (i['mean_error_valid'], i['st_dev_valid'],i['time']))
        best_configs = [config for config in gs_results[:10]]

    with open('NeuralNetwork/cup/best_config1.pickle', 'wb') as handle:
        pickle.dump(best_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
