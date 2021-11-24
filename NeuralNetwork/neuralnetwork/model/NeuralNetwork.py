import math
import numpy as np
from copy import deepcopy

class NeuralNetwork():
    def __init__(self, epochs=150, batch_size=64, lr=0.1, momentum=0.5, regularization='l2'):
        self.layers = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.regularization = regularization
        self.loss = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, y):
        self.X = X
        self.y = y
        

    def feedForward(self, batch):
        output = self.layers[0].forward(batch)
        for layer in self.layers[1:]:
            output = layer.forward(output)

        return output

    def train(self):

        for epoch in range(self.epochs):

            print("EPOCH {} --->".format(epoch))

            epoch_loss = []

            for it, n in enumerate(range(0,len(self.X),self.batch_size)):

                in_batch = self.X[n:n+self.batch_size]
                out_batch = self.y[n:n+self.batch_size]

                #FEEDFORWARD
                output = self.feedForward(in_batch)
                
                #BACKPROPAGATION
                error = out_batch - output
                for layer in reversed(self.layers):
                    error = layer.backward(error)

                
                #OPTIMIZATION
                for layer in self.layers:
                    layer.w_gradient /= self.batch_size
                    layer.b_gradient /= self.batch_size
                    delta_w = layer.w_gradient * self.lr
                    delta_b = layer.b_gradient * self.lr
                    layer.w_gradient = np.add(delta_w, layer.old_w_gradient*self.momentum)
                    layer.b_gradient = np.add(delta_b, layer.old_b_gradient*self.momentum)
                    layer.w = np.add(layer.w, layer.w_gradient)
                    layer.b = np.add(layer.b, layer.b_gradient)

                #batch loss
                mse = (np.sum((out_batch-output)**2))/self.batch_size
                print("{} ---> Loss:\t{}".format(it+1, mse))
                epoch_loss.append(mse)
            
            #epoch loss
            mean_loss = sum(epoch_loss)/ len(epoch_loss)
            self.loss.append(mean_loss)
            print("LOSS ---> {}\n".format(mean_loss))
            

    def predict(self, x_test):
        model = deepcopy(self)
        output = model.feedForward(x_test)
        return output
                
            

        
    #for n, layer in enumerate(self.layers):
    #    print("LAYER {}".format(n))
    #    print("INPUT\n",layer.input)
    #    print("GRADIENT\n",layer.gradient)

