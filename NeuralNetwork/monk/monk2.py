import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('neuralnetwork'))

import matplotlib.pyplot as plt
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_monk
from neuralnetwork.model.NeuralNetwork import build_model
from timeit import default_timer as timer



X_train, X_test, y_train, y_test = load_monk(1)

n_features = X_train.shape[1]
batch = len(X_train)

config = {
        'n_features': n_features,
        'n_hidden_layers':1,
        'n_units':4,
        'batch_size':batch,
        'out_units':1,
        'hidden_act':'tanh',
        'out_act':'sigmoid',
        'weights_init':'xavier_normal',
        'lr':0.9,
        'momentum':0.9,
        'reg_type':None,
        'lambda':0,
        'lr_decay':False,
        'nesterov':False
        }

results = {
        'train_loss': [],
        'valid_loss': [],
        'train_accuracy': [],
        'valid_accuracy': []
        }
        
plt.rcParams.update({'font.size':15})

for i in range(10):
        print("Iteration ---> ", i)
        
        #final model training and assessment
        model = build_model(config)
        model.compile('sgd',
                        loss='mse',
                        metric='accuracy',
                        lr=config['lr'],
                        momentum=config['momentum'],
                        reg_type=config['reg_type'],
                        lr_decay=config['lr_decay'],
                        nesterov=config['nesterov'],
                        lambd=config['lambda']
                        )

        model.fit(epochs=400,batch_size=config['batch_size'],X_train=X_train,y_train=y_train,X_valid=X_test,y_valid=y_test)
        results['train_loss'].append(model.history['train_loss'][-1]) 
        results['valid_loss'].append(model.history['valid_loss'][-1]) 
        results['train_accuracy'].append(model.history['train_accuracy'][-1]) 
        results['valid_accuracy'].append(model.history['valid_accuracy'][-1]) 

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,7))
        ax1.plot(model.history['train_loss'], label='Training')
        ax1.plot(model.history['valid_loss'], color='tab:orange', linestyle='dashed',label='Test')
        ax2.plot(model.history['train_accuracy'], label='Training')
        ax2.plot(model.history['valid_accuracy'], color='tab:orange', linestyle='dashed',label='Test')
        ax1.legend(fontsize=20)
        ax2.legend(fontsize=20)
        ax1.grid()
        ax2.grid()
        plt.savefig("NeuralNetwork/monk/plot/monk2_test_{}.png".format(i+1))

print("Train mse media: ", np.mean(results['train_loss']))
print("Test mse media: ", np.mean(results['valid_loss']))
print("Train accuracy media: ", np.mean(results['train_accuracy']))
print("Test accuracy media: ", np.mean(results['valid_accuracy']))
print("Train mse std: ", np.std(results['train_loss']))
print("Test mse std: ", np.std(results['valid_loss']))
print("Train accuracy std: ", np.std(results['train_accuracy']))
print("Test accuracy std: ", np.std(results['valid_accuracy']))



