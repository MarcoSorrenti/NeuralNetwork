from os import sep
import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_monk, load_cup
from neuralnetwork.model.Layer import Layer
from neuralnetwork.model.NeuralNetwork import NeuralNetwork, build_model
from neuralnetwork.model_selection import KFoldCV



#X_train, X_test, y_train, y_test = load_monk(1)
X_train, X_test, y_train, y_test = load_cup()

n_features = X_train.shape[1]
batch_size = len(X_train)

'''

model = NeuralNetwork()
input_layer = Layer(n_features,3, activation_function='relu', weights_init='random_init')
hidden_layer = Layer(3,1, activation_function='sigmoid', weights_init='xavier_uniform')
model.add_layer(input_layer)
model.add_layer(hidden_layer)

model.compile('sgd', loss='mse', metric='accuracy',lr=0.5,momentum=0.8,reg_type=None,lr_decay=False,nesterov=False)
es = EarlyStopping(10,1e-9)

model.fit(epochs=400,batch_size=batch_size,X_train=X_train, y_train=y_train, X_valid=X_test, y_valid=y_test, es=es)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))
ax1.plot(model.history['train_loss'], label='train_loss')
ax1.plot(model.history['valid_loss'], color='r', label='valid_loss')
ax2.plot(model.history['train_accuracy'], label='train_acc')
ax2.plot(model.history['valid_accuracy'], color='r', label='valid_acc')

ax1.legend()
ax2.legend()
plt.show()

'''


params = [{"n_units":30,"activation_function":'relu', "weights_init":'xavier_uniform'},
            {"n_units":30,"activation_function":'relu', "weights_init":'xavier_uniform'},
            {"n_units":30,"activation_function":'relu', "weights_init":'xavier_uniform'},
            {"n_units":2,"activation_function":'linear',"weights_init":'xavier_uniform'}]

NN = build_model(n_features=10, layers_params=params)
NN.compile('sgd', loss='mse', metric='mee',lr=0.001,momentum=0.9,reg_type=None,lr_decay=True,nesterov=False)
cv = KFoldCV(NN, X_train, y_train, 4, epochs=100, batch_size=batch_size)
#print(*((x,y) for x,y in cv._results.items()), sep='\n\n')







