import numpy as np
from neuralnetwork.regularization import regularization_dict
from neuralnetwork.utils.metrics import mse_loss, accuracy_bin
from neuralnetwork.lr_decay import Linear_decay
from neuralnetwork.utils.metrics import evaluation_metrics

class SGD:
    def __init__(self, model, loss='mse', eval_metric='accuracy', lr=0.1, momentum=0.5, nesterov=False, reg_type=None, lambd=None, lr_decay=False):

        self.model = model
        self.loss = evaluation_metrics[loss]
        self.eval_metric = evaluation_metrics[eval_metric]
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.regularization = regularization_dict[reg_type](lambd) if reg_type is not None else None
        self.lr_decay = Linear_decay(self.lr) if lr_decay else False


    def optimize(self, epochs, batch_size, X_train, y_train, X_valid=None, y_valid=None, es=False):

        self.train_loss = []
        self.train_accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []

        for epoch in range(epochs):

            print("EPOCH {}:".format(epoch))
            epoch_loss = []
            epoch_accuracy = []

            for it, n in enumerate(range(0,len(X_train),batch_size)):
                in_batch = X_train[n:n+batch_size]
                out_batch = y_train[n:n+batch_size]

                #nesterov momentum
                if self.nesterov:
                    for layer in self.model.layers:
                        layer.w = np.add(layer.w, self.momentum*layer.old_w_gradient)
                        layer.b = np.add(layer.b, self.momentum*layer.old_b_gradient)

                #gradient computation
                accuracy, mse = self.model.backprop(in_batch, out_batch)
                
                #OPTIMIZATION
                for layer in self.model.layers:

                    layer.w_gradient /= batch_size
                    layer.b_gradient /= batch_size
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
            

            #lr decay
            if self.lr_decay:
                self.lr = self.lr_decay.decay(epoch)

            #epoch evaluation
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            mean_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
            print("TRAINING || LOSS ---> {}\tACCURACY ---> {}".format(mean_loss, mean_accuracy))
            self.train_loss.append(mean_loss)
            self.train_accuracy.append(mean_accuracy)


            
            if X_valid is not None:

                #validation step
                y_pred_valid = self.model.predict(X_valid)
                mse_valid = self.loss(y_valid, y_pred_valid)
                acc_valid = self.eval_metric(y_valid, y_pred_valid)
                print("VALIDATION || LOSS ---> {}\tACCURACY ---> {}".format(mse_valid, acc_valid))

                #EARLY STOPPING
                if es and epoch > es.patience:
                    if not es.check_stopping(self, mse_valid):
                        self.model.history = {"train_loss":self.train_loss,"train_accuracy":self.train_accuracy,
                                                "valid_loss":self.valid_loss,"valid_accuracy":self.valid_accuracy}
                        print("ES: Training terminated.")
                        return

            #metrics update

                self.valid_loss.append(mse_valid)
                self.valid_accuracy.append(acc_valid)

            
            print("\n")
            

        self.model.history = {"train_loss":self.train_loss,"train_accuracy":self.train_accuracy,
                                "valid_loss":self.valid_loss,"valid_accuracy":self.valid_accuracy}




class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.tol = patience
        self.min_delta = min_delta

    def check_stopping(self, opt, t_monitor):

        gain = opt.valid_loss[-1] - t_monitor
        if gain < self.min_delta and self.tol > 0:
            self.tol -= 1
            print("ES: No improvement")
        else:
            self.tol = self.patience
            
        if self.tol == 0:
            return 0

        return 1


optimizers = {'sgd':SGD}