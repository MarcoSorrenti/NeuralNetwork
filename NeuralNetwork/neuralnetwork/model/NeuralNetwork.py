import math
import numpy as np
from copy import deepcopy
from neuralnetwork.regularization import regularization_dict
from neuralnetwork.utils.metrics import accuracy_bin, mse_loss

class NeuralNetwork():
    def __init__(self, epochs=150, batch_size=64, lr=0.1, momentum=0.5, regularization=None, lambd=0.001):
        self.layers = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.regularization = regularization_dict[regularization](lambd) if regularization is not None else None
        self.loss = []
        self.accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []

    def add_layer(self, layer):
        self.layers.append(layer)

    

    def feedForward(self, output):
        penalty_term = 0
        for layer in self.layers:           
            output = layer.forward(output)
            if self.regularization is not None:
                penalty_term += self.regularization.compute(layer.w)
        return output, penalty_term


    def backprop(self, input, y_true):
        
        y_pred, penalty_term = self.feedForward(input)
        print(penalty_term)

        error = y_true - y_pred

        mse = mse_loss(y_true, y_pred) + penalty_term
        acc = accuracy_bin(y_true, y_pred)

        for layer in reversed(self.layers):
            error = layer.backward(error)

        return acc, mse


    def fit(self, X_train, y_train, X_valid=None, y_valid=None):

        X_train = X_train
        y_train = y_train

        for epoch in range(self.epochs):
            print("EPOCH {}:".format(epoch))
            epoch_loss = []
            epoch_accuracy = []

            for it, n in enumerate(range(0,len(X_train),self.batch_size)):
                in_batch = X_train[n:n+self.batch_size]
                out_batch = y_train[n:n+self.batch_size]

                accuracy, mse = self.backprop(in_batch, out_batch)
                
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



                #batch evaluation
                print("{} \\\\ Loss:\t{}\tAccuracy:\t{}".format(it + 1, mse, accuracy))
                epoch_loss.append(mse)
                epoch_accuracy.append(accuracy)
            


            #epoch evaluation
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            mean_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
            self.loss.append(mean_loss)
            self.accuracy.append(mean_accuracy)
            print("LOSS ---> {}\tACCURACY ---> {}".format(mean_loss, mean_accuracy))


            y_pred_valid = self.predict(X_valid)
            mse_valid = mse_loss(y_valid, y_pred_valid)
            acc_valid = accuracy_bin(y_valid, y_pred_valid)
            self.valid_loss.append(mse_valid)
            self.valid_accuracy.append(acc_valid)




    def predict(self, x_test):
        model = deepcopy(self)
        output, _ = model.feedForward(x_test)
        return output

