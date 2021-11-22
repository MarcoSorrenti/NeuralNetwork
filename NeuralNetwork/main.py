import sys
from neuralnetwork.datasets.util import import_dataset
from neuralnetwork.model.Layer import Layer
from neuralnetwork.model.NeuralNetwork import NeuralNetwork


X_train, X_test, y_train, y_test = import_dataset(1)

n_features = X_train.shape[1]


model = NeuralNetwork()
input_layer = Layer(n_features,5)
hidden_layer = Layer(5,1)
model.add_layer(input_layer)
model.add_layer(hidden_layer)
model.fit(X_train, y_train) 
model.train()
