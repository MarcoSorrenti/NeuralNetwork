import pandas as pd
import numpy as np
from FeedForwardNN import NeuralNetwork
from Layers import DLayer
import matplotlib.pyplot as plt

    
MONK_PROBLEM = 1

#TRAINING SET
monk_train = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-{}.train".format(MONK_PROBLEM),
    header=None,
    sep=' ',
    skipinitialspace=True
    )

#TRAINING SET PREPROCESSING
monk_train = monk_train.iloc[:,:-1]
monk_train = monk_train.sample(frac=1, random_state=42)

X_train_to_encode = monk_train.iloc[:,1:]
#one hot encoding
X_train = pd.get_dummies(X_train_to_encode, columns=[col for col in X_train_to_encode.columns])
y_train = monk_train.iloc[:,0]
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1,1)
print("Training set shape: {X_train.shape}")

#TEST SET
monk_test = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-{}.test".format(MONK_PROBLEM),
    header=None,
    sep=' ',
    skipinitialspace=True
    )

#TEST SET PREPROCESSING
monk_test = monk_test.iloc[:,:-1]
monk_test = monk_test.sample(frac=1, random_state=42)

X_test_to_encode = monk_test.iloc[:,1:]
#one hot encoding
X_test = pd.get_dummies(X_test_to_encode, columns=[col for col in X_test_to_encode.columns])
y_test = monk_test.iloc[:,0]
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1,1)
print("Test set shape: {X_test.shape}")

b_size = X_train.shape[0]
n_features = X_train.shape[1]

model = NeuralNetwork(X_train, y_train)
input_layer = DLayer(n_features,3,'tanh','xavier_uniform')
hidden1 = DLayer(3,1,'tanh','xavier_uniform')
model.add(input_layer)
model.add(hidden1)

model.train(lr=0.2,
            epochs=200,
            batch_size=b_size,
            momentum=0.8,
            loss_function='mse',
            lambd=0.,
            lr_factor=0.)


#MONK-1 {tanh, tanh, lr:0.2, momentum:0.8, epochs:200}
#MONK-2 {tanh, tanh, lr:0.3, momentum:0.7, epochs:200}
#MONK-3 {lr:0.15, momentum:0.9, epochs:150/200, lambda: 0.005}


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
ax1.plot(model.loss)
ax1.set_title('LOSS')
ax2.plot(model.accuracy)
ax2.set_title('ACCURACY')
plt.show()


