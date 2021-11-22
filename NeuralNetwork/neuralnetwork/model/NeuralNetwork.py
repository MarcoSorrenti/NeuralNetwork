import numpy as np

class NeuralNetwork():
    def __init__(self, epochs=200, batch=128, lr=0.01, momentum=0.5, regularization='l2'):
        self.layers = []
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.momentum = momentum
        self.regularization = regularization

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, y):
        self.X = X
        self.y = y 

    def feedForward(self):
        output = self.layers[0].forward(self.X)
        for layer in self.layers[1:]:
            output = layer.forward(output)

        return output

    def train(self):
        output = self.feedForward()
        error = self.y - output
        mse = (np.sum((error)**2))/len(self.y)
        
        delta = self.layers[-1].backward(error)
        for layer in reversed(range(len(self.layers[:-1]))):
            #computation of delta_j added inside the backward function
            #delta = np.dot(delta,self.layers[layer-1].w.T) 
            self.layers[layer].backward(delta)

        for layer in self.layers:
            print(layer.gradient)

