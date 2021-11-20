import sys
from neuralnetwork.datasets.util import import_dataset
from neuralnetwork.model.Layer import Layer

X_train, X_test, y_train, y_test = import_dataset(3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

n_features = X_train.shape[1]
layer = Layer(n_features, 3)

layer.forward(X_train[0])
print(layer.output)