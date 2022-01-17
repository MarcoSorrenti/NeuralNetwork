import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_monk, load_cup
from neuralnetwork.model.Layer import Layer
from neuralnetwork.model.NeuralNetwork import NeuralNetwork, build_model
from neuralnetwork.model_selection import KFoldCV, GridSearchCVNN



import warnings
warnings.filterwarnings("ignore")

#X_train, X_test, y_train, y_test = load_monk(1)
X_train, X_test, y_train, y_test = load_cup()

n_features = X_train.shape[1]
batch = len(X_train)

if not os.path.isfile('./best_config.pickle'):

    params_config = {
                'n_features': [n_features],
                'n_hidden_layers':[3,5],
                'n_units':[10,20,30],
                'batch_size':[128, 256],
                'out_units':[2],
                'hidden_act':['relu'],
                'out_act':['linear'],
                'weights_init':['xavier_uniform','he_normal','he_uniform'],
                'lr':[0.001, 0.005],
                'momentum':[0.9],
                'reg_type': ['l2'],
                'lr_decay':[False],
                'nesterov':[False]
                }

    gs = GridSearchCVNN(params_config)
    gs.fit(X_train,y_train,loss='mee',epochs=150,shuffle=True)
    gs_results = sorted(gs.grid_results, key = lambda i: (i['mean_error_valid'], i['st_dev_valid'],i['time']))
    best_config = gs_results[0]['parameters']

    with open('best_config.pickle', 'wb') as handle:
        pickle.dump(best_config, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('best_config.pickle', 'rb') as handle:
    b = pickle.load(handle)


model = build_model(best_config)
model.compile('sgd',
                loss='mse',
                lr=best_config['lr'],
                momentum=best_config['momentum'],
                reg_type=best_config['reg_type'],
                lr_decay=best_config['lr_decay'],
                nesterov=best_config['nesterov'])

es = EarlyStopping(10,1e-7)
model.fit(epochs=500,batch_size=best_config['batch_size'],X_train=X_train,y_train=y_train,X_valid=X_test,y_valid=y_test)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))
ax1.plot(model.history['train_loss'], label='train_loss')
ax1.plot(model.history['valid_loss'], color='r', label='valid_loss')
ax2.plot(model.history['train_accuracy'], label='train_acc')
ax2.plot(model.history['valid_accuracy'], color='r', label='valid_acc')

ax1.legend()
ax2.legend()
plt.show()
