import os, pickle, pathlib
import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_monk, load_cup, train_test_split
from neuralnetwork.model.Layer import Layer
from neuralnetwork.model.NeuralNetwork import NeuralNetwork, build_model
from neuralnetwork.model_selection import KFoldCV, GridSearchCVNN


#X_train, X_test, y_train, y_test = load_monk(1)
X_train, X_test_blind, y_train, y_test_blind = load_cup()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

n_features = X_train.shape[1]
batch = len(X_train)

run_grid = False

if not os.path.isfile('NeuralNetwork/best_config.pickle') or run_grid:

    params_config = {
                'n_features': [n_features],
                'n_hidden_layers':[3,5],
                'n_units':[10,20,30],
                'batch_size':[128, 256],
                'out_units':[2],
                'hidden_act':['relu'],
                'out_act':['linear'],
                'weights_init':['xavier_uniform','he_normal','he_uniform'],
                'lr':[0.001, 0.003, 0.005],
                'momentum':[0.9],
                'reg_type': ['l2'],
                'lambda':[0.001, 0.01],
                'lr_decay':[False],
                'nesterov':[False]
                }

    gs = GridSearchCVNN(params_config)
    gs.fit(X_train,y_train,loss='mee',epochs=150,shuffle=True)
    gs_results = sorted(gs.grid_results, key = lambda i: (i['mean_error_valid'], i['st_dev_valid'],i['time']))
    best_config = gs_results[0]['parameters']

    with open('NeuralNetwork/best_config.pickle', 'wb') as handle:
        pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('NeuralNetwork/best_config.pickle', 'rb') as handle:
    best_config = pickle.load(handle)


model = build_model(best_config)
model.summary()
model.compile('sgd',
                loss='mee',
                lr=best_config['lr'],
                momentum=best_config['momentum'],
                reg_type=best_config['reg_type'],
                lr_decay=best_config['lr_decay'],
                nesterov=best_config['nesterov'],
                lambd=best_config['lambda'])

#es = EarlyStopping(10,1e-15)
model.fit(epochs=200,batch_size=best_config['batch_size'],X_train=X_train,y_train=y_train,X_valid=X_test,y_valid=y_test)

plt.figure(figsize=(15,7))
plt.plot(model.history['train_loss'], label='train_loss')
plt.plot(model.history['valid_loss'], color='r', label='valid_loss')
plt.legend()
plt.show()

