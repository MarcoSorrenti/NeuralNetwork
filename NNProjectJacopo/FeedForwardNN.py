import numpy as np
from copy import deepcopy
from loss_functions import losses_dict
from activation_functions import sign
from evaluation_metrics import accuracy_metric
from regularizations_weights_decay import *

#neural network structure
class NeuralNetwork(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = len(y)
        self.layers = list()
        self.loss = list()
        self.accuracy = list()

        
    def add(self, layer):
        self.layers.append(layer) 
        

    def train(self, lr=0.1, epochs=200, batch_size=64, momentum=0.3, loss_function='binary_crossentropy', lambd=0., lr_factor=None):
        
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        layers = self.layers

        
        for epoch in range(epochs):
            
            print("EPOCH {}:".format(epoch+1))
            
            #implement shuffle only in mini-batch mode
            #self.X, self.y = double_shuffle(self.X, self.y)
            
            
            #forward pass
            inputs = self.X
            for i in range(len(layers)):
                inputs = layers[i].forward(inputs)

            #calculating error
            error = -(self.y - layers[-1].outputs)

            #BACKWARD PASS --> computing gradients
            for n in reversed(range(len(layers))):
                error = layers[n].backward(error)


            for l in range(len(layers)):

                #correcting gradients based on batch
                layers[l].weight_gradients /= self.N
                layers[l].bias_gradients /= self.N
                
                #introducing indipendent delta term with learning rate
                delta_w = layers[l].weight_gradients * self.lr
                delta_b = layers[l].bias_gradients * self.lr

                #applying momentum to old weights gradients
                layers[l].old_weight_gradients *= self.momentum
                layers[l].old_bias_gradients *= self.momentum

                #FINAL DELTA including old weight gradient whether MOMENTUM is different from 0
                layers[l].weight_gradients = np.add(delta_w,
                                                    layers[l].old_weight_gradients)
                layers[l].bias_gradients = np.add(delta_b, layers[l].old_bias_gradients)

                #PARAMETERS UPDATE
                layers[l].weights = np.add(layers[l].weights, np.add(layers[l].weight_gradients, -lambd*layers[l].weights)) # regularization
                layers[l].biases = np.add(layers[l].biases, layers[l].bias_gradients)

                
            y_pred = layers[-1].outputs
            loss = losses_dict[loss_function](y_pred, self.y)
            accuracy = accuracy_metric(self.y, sign(y_pred))
            self.loss.append(loss)
            self.accuracy.append(accuracy)
            print("\tLoss: {}\t Accuracy: {}".format( round(loss,8), round(accuracy,8) ))

            #learning rate decay --> implement as a class
            self.lr = lr_decay(epoch, self.lr, lr_factor)
            
            #stopping criteria


            
    def predict(self, input_data):
        
        predictive_model = deepcopy(self)
        N = len(input_data)
        
        predicts = input_data
        for layer in predictive_model.layers:
            predicts = layer.forward(predicts)
        
        return predicts
    
    
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)
        
        return np.array(weights)
    
    