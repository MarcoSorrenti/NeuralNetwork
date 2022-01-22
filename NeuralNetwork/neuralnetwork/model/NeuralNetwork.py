import numpy as np
import matplotlib.pyplot as plt 
from copy import deepcopy
from neuralnetwork.model.Optimizer import optimizers
from neuralnetwork.model.Layer import Layer
from neuralnetwork.utils.metrics import evaluation_metrics

class NeuralNetwork():
    '''Neural network model class.'''
    def __init__(self):
        '''Constructor
        layers: layers array structure initialization
        history: dictionary initialization for training memorization of loss and evaluation metrics 
        '''
        self.layers = []
        self.history = dict()

    def add_layer(self, layer):
        '''Function for layer addition to the network structure'''
        self.layers.append(layer)

    def feedForward(self, output):
        '''Feed-forward propagation function. 
        Propagates the input through layers, also computing the penalty term if regularization is needed
        Args:
            output : actual input to propagate forward from the input layer  
        '''
        penalty_term = 0
        for layer in self.layers:           
            output = layer.forward(output)
            if self.optimizer.regularization is not None:
                penalty_term += self.optimizer.regularization.compute(layer.w)
        return output, penalty_term

    def backprop(self, input, y_true):
        '''Backpropagation function.
        Args:
            input:
            y_true:
        Returns:
            eval_metric:
            mse:
        '''
        y_pred, penalty_term = self.feedForward(input)
        error = y_true - y_pred
        mse = self.optimizer.loss(y_true, y_pred) + penalty_term
        eval_metric = self.optimizer.eval_metric(y_true, y_pred) if self.optimizer.eval_metric else None

        for layer in reversed(self.layers):
            error = layer.backward(error)

        return eval_metric, mse


    def compile(self, opt='sgd', loss='mse', metric=None, lr=0.1, momentum=0.5, nesterov=False, reg_type=None, lambd=0.001, lr_decay=False):
        '''Compile function.
        Args:
            opt: optimizer 
            loss:
            metric: 
            lr: learning rate 
            momentum: momentum 
            nesterov: 
            reg_type: regularization 
            lambd: lambda 
            lr_decay: learning rate decay
        '''
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

    def fit(self, epochs=200, batch_size=128, X_train=None, y_train=None, X_valid=None, y_valid=None, es=False):
        '''Fit function.
        Args:
            epochs: # of epochs 
            batch_size:
            X_train
            y_train
            X_valid
            y_valid
            es: early stopping object
        Returns:
            history
        '''
        self.optimizer.optimize(epochs=epochs, batch_size=batch_size, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, es=es)
        return self.history

    def predict(self, x_test):
        '''Predict function.
        Args:
            X_test: input data
        Returns:
            output: prediction
        '''
        model = deepcopy(self)
        output, _ = model.feedForward(x_test)
        return output

    def evaluate(self, x_test, y_test, metrics=['mse']):
        '''Evaluate function.
        Args:
           x_test
           y_test
           metrics
        Returns:
            scores
        '''
        y_preds = self.predict(x_test)
        scores = dict()
        for metric in metrics:
            scores[metric] = evaluation_metrics[metric](y_test,y_preds)
        return scores

    def reset_model(self):
        '''Reset model function.'''
        for layer in self.layers:
            layer.set_parameters()
        self.optimizer = self.backup_opt
        self.history = dict()

    def summary(self):
        '''Summary function.'''
        print(" MODEL ".center(35,' '))
        print("-"*35)
        print("{: >21}{: >11}".format("Input","Output"))
        print("-"*35)

        for i,layer in enumerate(self.layers):
            print("Layer_{}: {: >10} {: >10}".format(i, layer.input_dim, layer.n_units), sep='\t')
            print("—"*35)

    def copy_model(self):
        '''Copy model function.
        Returns:
            deepcopy of the model
        '''
        return deepcopy(self)

    def plot_metrics(self, show=False,save_path=None):
        '''Plot metrics function.
        Args:
           show: boolean. Show plots if True.
           save_path: string. Save plots if given
        '''
        plt.rcParams.update({'font.size':15})
        if self.optimizer.eval_metric != None:
            eval_metric = self.optimizer.eval_metric_text
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,7))
            ax1.plot(self.history['train_loss'], label='Training')
            ax1.plot(self.history['valid_loss'], color='tab:orange', linestyle='dashed',label='Test')
            ax2.plot(self.history['train_{}'.format(eval_metric)], label='Training')
            ax2.plot(self.history['valid_{}'.format(eval_metric)], color='tab:orange', linestyle='dashed',label='Test')
            ax1.legend(fontsize=20)
            ax2.legend(fontsize=20)
            ax1.grid()
            ax2.grid()        
        else:
            plt.plot(self.history['train_loss'], label='Training')
            plt.plot(self.history['valid_loss'], color='tab:orange', linestyle='dashed',label='Test')
            plt.legend(fontsize=20)
            plt.grid()

        if save_path:
            plt.savefig(save_path)

        if show == True:
            plt.show()


#model building function
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
