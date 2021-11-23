import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork.datasets.util import load_monk
from neuralnetwork.model.Layer import Layer
from neuralnetwork.model.NeuralNetwork import NeuralNetwork


X_train, X_test, y_train, y_test = load_monk(2)

n_features = X_train.shape[1]

batch_size = len(X_train)
model = NeuralNetwork(epochs=200,batch_size=batch_size)
input_layer = Layer(n_features,2)
hidden_layer = Layer(2,1)
model.add_layer(input_layer)
model.add_layer(hidden_layer)
model.fit(X_train, y_train) 
model.train()

#plt.plot(model.loss)
#plt.show()

y_preds = model.predict(X_test)

error = y_test - y_preds
mse = (np.sum((error)**2))/len(y_preds)
print(mse)
print(y_preds)
print(y_test)

