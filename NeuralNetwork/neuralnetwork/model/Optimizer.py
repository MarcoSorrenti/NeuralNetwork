import numpy as np
from neuralnetwork.regularization import regularization_dict

class Optimizer:
    def __init__(self, model, loss, metric, lr=0.1, momentum=0.5, reg_type=None, lambd=None):
        self.model = model
        self.loss = loss
        self.metric = metric
        self.lr = lr
        self.momentum = momentum
        self.regularization = regularization_dict[reg_type](lambd)

    

    def optimize(self, X_train, y_train, epochs, batch_size):

        for epoch in range(epochs):
            print("EPOCH {} --->".format(epoch))
            epoch_loss = []

            for it, n in enumerate(range(0,len(self.X),batch_size)):
                in_batch = X_train[n:n+batch_size]
                out_batch = y_train[n:n+batch_size]

                mse = self.model.backprop(in_batch, out_batch,reg=self.reg_type)
                
                #OPTIMIZATION
                for layer in self.model.layers:

                    layer.w_gradient /= batch_size
                    layer.b_gradient /= batch_size
                    delta_w = layer.w_gradient * self.lr
                    delta_b = layer.b_gradient * self.lr
                    layer.w_gradient = np.add(delta_w, layer.old_w_gradient*self.momentum)
                    layer.b_gradient = np.add(delta_b, layer.old_b_gradient*self.momentum)
                    layer.w = np.add(layer.w, layer.w_gradient)

                    if self.regularization is not None:
                        layer.w = np.subtract(layer.w,self.regularization.derivate(layer.w))
                        
                    layer.b = np.add(layer.b, layer.b_gradient)

                #batch loss
                print("{} ---> Loss:\t{}".format(it + 1, mse))
                epoch_loss.append(mse)
            
            #epoch loss
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            self.loss.append(mean_loss)
            print("LOSS ---> {}\n".format(mean_loss))