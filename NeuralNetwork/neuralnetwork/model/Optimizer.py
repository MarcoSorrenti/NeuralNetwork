from copy import deepcopy
import numpy as np
from neuralnetwork.utils.regularization import regularization_dict
from neuralnetwork.utils.metrics import mse_loss, accuracy_bin
from neuralnetwork.utils.lr_decay import Linear_decay
from neuralnetwork.utils.metrics import evaluation_metrics

class SGD:
    def __init__(self, model, loss='mse', eval_metric=None, lr=0.1, momentum=0.5, nesterov=False, reg_type=None, lambd=None, lr_decay=False):

        self.model = model
        self.loss = evaluation_metrics[loss]
        self.loss_text = loss
        self.eval_metric = evaluation_metrics[eval_metric] if eval_metric else None
        self.eval_metric_text = eval_metric
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.regularization = regularization_dict[reg_type](lambd) if reg_type is not None else None
        self.lr_decay = Linear_decay(self.lr) if lr_decay else False

    def optimize(self, epochs, batch_size, X_train, y_train, X_valid=None, y_valid=None, es=False):

        self.history = {
            'train_loss':list(),
            'valid_loss':list()
        }

        if self.eval_metric:
            self.history.update({'train_{}'.format(self.eval_metric_text):list(),
                                    'valid_{}'.format(self.eval_metric_text):list()})

        for epoch in range(epochs):

            print("EPOCH {}:".format(epoch))
            epoch_loss = []
            epoch_accuracy = []

            if batch_size < len(X_train):

                perm = np.random.permutation(len(X_train))
                X_train, y_train = X_train[perm], y_train[perm]


            for it, n in enumerate(range(0,len(X_train),batch_size)):
                in_batch = X_train[n:n+batch_size]
                out_batch = y_train[n:n+batch_size]


                #nesterov momentum --> velocity
                if self.nesterov:
                    for layer in self.model.layers:
                        layer._nesterov_w = deepcopy(layer.w)
                        layer._nesterov_b = deepcopy(layer.b)
                        layer.w = np.add(layer.w, self.momentum*layer.old_w_gradient)
                        layer.b = np.add(layer.b, self.momentum*layer.old_b_gradient)


                #gradient computation
                eval_metric, loss = self.model.backprop(in_batch, out_batch)

                #OPTIMIZATION
                for layer in self.model.layers:
                    if self.nesterov:
                        layer.w = layer._nesterov_w
                        layer.b = layer._nesterov_b

                    delta_w = layer.w_gradient * self.lr / batch_size
                    delta_b = layer.b_gradient * self.lr / batch_size

                    #regular momentum step
                    layer.w_gradient = np.add(delta_w, layer.old_w_gradient*self.momentum)
                    layer.b_gradient = np.add(delta_b, layer.old_b_gradient*self.momentum)

                    if self.regularization is not None:

                        layer.w = np.add(layer.w, layer.w_gradient - self.regularization.derivate(layer.w))
                    else:
                        layer.w = np.add(layer.w, layer.w_gradient)

                    layer.b = np.add(layer.b, layer.b_gradient)


                #batch evaluation
                if batch_size < X_train.shape[0]:
                    print("{} \\\\ Loss:\t{}".format(it + 1, loss), end="\n")

                    if self.eval_metric : print("{}:\t{}".format(self.eval_metric_text.title(), eval_metric))

                epoch_loss.append(loss)
                if self.eval_metric:
                    epoch_accuracy.append(eval_metric)
            

            #lr decay
            if self.lr_decay:
                self.lr = self.lr_decay.decay(epoch)

            #epoch evaluation
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            self.history['train_loss'].append(mean_loss)
            print("TRAINING || Loss ---> {}".format(mean_loss), end="    ")

            if self.eval_metric :
                mean_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
                self.history['train_{}'.format(self.eval_metric_text)].append(mean_accuracy)
                print("{} ---> {}".format(self.eval_metric_text.title(), mean_accuracy))
                
            
            
            if X_valid is not None:

                #validation step
                y_pred_valid = self.model.predict(X_valid)
                loss_valid = self.loss(y_valid, y_pred_valid)
                self.history['valid_loss'].append(loss_valid)
                
                print("VALIDATION || Loss ---> {}".format(loss_valid),end="    ")

                if self.eval_metric : 
                    eval_metric_valid = self.eval_metric(y_valid, y_pred_valid)
                    self.history['valid_{}'.format(self.eval_metric_text)].append(eval_metric_valid)
                    print("{} ---> {}".format(self.eval_metric_text.title(), eval_metric_valid))

                #EARLY STOPPING
                if es and epoch > es.patience:
                    if not es.check_stopping(self):

                        self.model.history = self.history

                        return

            print("\r") 

        self.model.history = self.history



class EarlyStopping:
    '''Class for Training interruption based on a monitoring metric to check on every epoch.
    Parameters:
        monitor : metric to monitor
        patience : tolerance epochs
    '''
    def __init__(self, monitor='valid_loss',patience=10, min_delta=1e-23):
        self.monitor = monitor
        self.patience = patience
        self.tol = patience
        self.min_delta = min_delta

    def __repr__(self):
        return "ES: {} patience, {} min_delta".format(self.monitor, self.patience, self.min_delta)


    def check_stopping(self, opt):

        gain = np.min(opt.history[self.monitor][:-1]) - opt.history[self.monitor][-1] 
        if gain < self.min_delta and self.tol > 0:
            self.tol -= 1
            print("\nES: No improvement",end='\r')
        else:
            self.tol = self.patience
            
        if self.tol == 0:
            print("\nES: TRAINING TERMINATED.")
            self.tol = self.patience
            return 0

        return 1


optimizers = {'sgd':SGD}