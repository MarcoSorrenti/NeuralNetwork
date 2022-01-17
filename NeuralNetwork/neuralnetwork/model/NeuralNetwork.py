import numpy as np
from copy import deepcopy
from neuralnetwork.utils.metrics import accuracy_bin, mse_loss
from neuralnetwork.model.Optimizer import optimizers
from neuralnetwork.model.Layer import Layer

class NeuralNetwork():
    def __init__(self):
        self.layers = []
        self.history = dict()

    def add_layer(self, layer):
        self.layers.append(layer)
    

    def feedForward(self, output):
        penalty_term = 0
        for layer in self.layers:           
            output = layer.forward(output)
            if self.optimizer.regularization is not None:
                penalty_term += self.optimizer.regularization.compute(layer.w)
        return output, penalty_term


    def backprop(self, input, y_true):
        y_pred, penalty_term = self.feedForward(input)
        error = y_true - y_pred
        mse = self.optimizer.loss(y_true, y_pred) + penalty_term
        eval_metric = self.optimizer.eval_metric(y_true, y_pred) if self.optimizer.eval_metric else None

        for layer in reversed(self.layers):
            error = layer.backward(error)

        return eval_metric, mse


    def compile(self, opt='sgd', loss='mse', metric=None, lr=0.1, momentum=0.5, nesterov=False, reg_type=None, lambd=0.001, lr_decay=False):

        self.optimizer = optimizers[opt](model=self, 
                                    loss=loss,
                                    eval_metric=metric,
                                    lr=lr,
                                    momentum=momentum, 
                                    nesterov=nesterov, 
                                    reg_type=reg_type, 
                                    lambd=lambd, 
                                    lr_decay=lr_decay)

        self.backup_opt = self.optimizer

        #model autosave before fit for cv
        #self._compiled_model = self.copy_model()


    def fit(self, epochs=200, batch_size=128, X_train=None, y_train=None, X_valid=None, y_valid=None, es=False):
        
        self.optimizer.optimize(epochs=epochs, batch_size=batch_size, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, es=es)
        
        return self.history


    def predict(self, x_test):
        model = deepcopy(self)
        output, _ = model.feedForward(x_test)
        return output



    def reset_model(self):
        for layer in self.layers:
            layer.set_parameters()

        self.optimizer = self.backup_opt

        self.history = dict()


    def summary(self):

        print(" MODEL ".center(35,' '))
        print("-"*35)
        print("{: >21}{: >11}".format("Input","Output"))
        print("-"*35)
        for i,layer in enumerate(self.layers):
            print("Layer_{}: {: >10} {: >10}".format(i, layer.input_dim, layer.n_units), sep='\t')
            print("—"*35)


    def copy_model(self):
        return deepcopy(self)





#model building

def build_model(layers_params:dict):
    '''Neural Network builder class

    Args:
        n_features: number of input features.
        layers_params: list of layers, where a layer is a dictionary of Layer constructor's parameters
    Returns:
        model: NN class model, ready to be compiled and fitted
    
    '''
    model = NeuralNetwork()

    #input layer
    input_layer = Layer(layers_params['n_features'], 
                        layers_params['n_units'], 
                        activation_function=layers_params['hidden_act'], 
                        weights_init=layers_params['weights_init'])
    model.add_layer(input_layer)
    in_units = input_layer.n_units

    #hidden layers
    for i in range(layers_params['n_hidden_layers']):
        layer = Layer(in_units, 
                        layers_params['n_units'], 
                        activation_function=layers_params['hidden_act'],
                        weights_init=layers_params['weights_init'])
        in_units = layer.n_units
        model.add_layer(layer)

    #output_layer
    layer = Layer(in_units, 
                    layers_params['out_units'], 
                    activation_function=layers_params['out_act'],
                    weights_init=layers_params['weights_init'])
    in_units = layer.n_units
    model.add_layer(layer)

    return model
