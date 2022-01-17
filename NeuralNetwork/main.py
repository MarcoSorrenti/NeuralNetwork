import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_monk, load_cup
from neuralnetwork.model.Layer import Layer
from neuralnetwork.model.NeuralNetwork import NeuralNetwork, build_model
from neuralnetwork.model_selection import KFoldCV, GridSearchCVNN

#X_train, X_test, y_train, y_test = load_monk(1)
X_train, X_test, y_train, y_test = load_cup()

n_features = X_train.shape[1]
batch = len(X_train)
'''
params = [{"n_units":30,"activation_function":'relu', "weights_init":'xavier_uniform'},
            {"n_units":30,"activation_function":'relu', "weights_init":'xavier_uniform'},
            {"n_units":30,"activation_function":'relu', "weights_init":'xavier_uniform'},
            {"n_units":2,"activation_function":'linear',"weights_init":'xavier_uniform'}]

NN = build_model(n_features=n_features, layers_params=params)
NN.compile('sgd', loss='mse', metric='accuracy', lr=0.001,momentum=0.9,reg_type=None,lr_decay=True,nesterov=False)
cv = KFoldCV(NN, X_train, y_train, 4, epochs=100, batch_size=batch_size)'''
#print(*((x,y) for x,y in cv._results.items()), sep='\n\n')

'''
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))
ax1.plot(model.history['train_loss'], label='train_loss')
ax1.plot(model.history['valid_loss'], color='r', label='valid_loss')
ax2.plot(model.history['train_accuracy'], label='train_acc')
ax2.plot(model.history['valid_accuracy'], color='r', label='valid_acc')

ax1.legend()
ax2.legend()
plt.show()
'''

params_config = {
            'n_features': [n_features],
            'n_hidden_layers':[3,4,5],
            'n_units':[10,20,30],
            'batch_size':[64, 128, 256],
            'out_units':[2],
            'hidden_act':['relu'],
            'out_act':['linear'],
            'weights_init':['xavier_uniform'],
            'lr':[0.001, 0.003],
            'momentum':[0.9],
            'reg_type': ['l2'],
            'lr_decay':[True],
            'nesterov':[False]
            }

gs = GridSearchCVNN(params_config)
gs.fit(X_train,y_train,loss='mee',epochs=150,shuffle=True)
print(*gs.grid_results, sep='\n')












