import math
import numpy as np
from copy import deepcopy
from neuralnetwork.regularization import regularization_dict

class NeuralNetwork():
    def __init__(self, epochs=150, batch_size=64, lr=0.1, momentum=0.5, regularization=None, lambd=0.001):
        self.layers = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.regularization = regularization_dict[regularization](lambd)
        self.loss = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def feedForward(self, output):
        penalty_term = 0
        for layer in self.layers:           
            output = layer.forward(output)
            if self.regularization is not None:
                penalty_term += self.regularization.compute(layer.w)
        return output, penalty_term


    def backprop(self, input, y_true, reg=False):
        y_pred, penalty_term = self.feedForward(input)
        if reg is None: penalty_term = 0

        error = y_true - y_pred
        mse = (np.sum((error)**2) + penalty_term) / self.batch_size

        for layer in reversed(self.layers):
            error = layer.backward(error)
            
        return mse

    def train(self):

        for epoch in range(self.epochs):
            print("EPOCH {} --->".format(epoch))
            epoch_loss = []

            for it, n in enumerate(range(0,len(self.X),self.batch_size)):
                in_batch = self.X[n:n+self.batch_size]
                out_batch = self.y[n:n+self.batch_size]

                mse = self.backprop(in_batch, out_batch)
                
                #OPTIMIZATION
                for layer in self.layers:
                    layer.w_gradient /= self.batch_size
                    layer.b_gradient /= self.batch_size
                    delta_w = layer.w_gradient * self.lr
                    delta_b = layer.b_gradient * self.lr
                    layer.w_gradient = np.add(delta_w, layer.old_w_gradient*self.momentum)
                    layer.b_gradient = np.add(delta_b, layer.old_b_gradient*self.momentum)
                    if self.regularization is not None:
                        layer.w = np.add(layer.w, layer.w_gradient - self.regularization.derivate(layer.w))
                    else:
                        layer.w = np.add(layer.w, layer.w_gradient)
                    layer.b = np.add(layer.b, layer.b_gradient)

                #batch loss
                print("{} ---> Loss:\t{}".format(it + 1, mse))
                epoch_loss.append(mse)
            
            #epoch loss
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            self.loss.append(mean_loss)
            print("LOSS ---> {}\n".format(mean_loss))
    

    def predict(self, x_test):
        model = deepcopy(self)
        output, _ = model.feedForward(x_test)
        return output

