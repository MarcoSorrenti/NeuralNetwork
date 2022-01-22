import os
import sys

sys.path.insert(0, os.path.abspath('neuralnetwork'))

import pickle
import numpy as np
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_cup, train_test_split
from neuralnetwork.model.NeuralNetwork import build_model
from timeit import default_timer as timer

X_train, X_test_blind, y_train, y_test_blind = load_cup()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

X_train_lvo, X_valid_lvo, y_train_lvo, y_valid_lvo = train_test_split(X_train, y_train)

n_features = X_train.shape[1]
batch = len(X_train)

#chosing grid search best configuration
with open('NeuralNetwork/cup/best_config1.pickle', 'rb') as handle:
    gs_best_configs = pickle.load(handle)
    
best_config = gs_best_configs[0]['parameters']

#model retraining and assessment

models = list()
times = list()
trials = 5
for trial in range(trials):

    model = build_model(best_config)
    model.compile('sgd',
                    loss='mee',
                    lr=best_config['lr'],
                    momentum=best_config['momentum'],
                    reg_type=best_config['reg_type'],
                    lr_decay=best_config['lr_decay'],
                    nesterov=best_config['nesterov'],
                    lambd=best_config['lambda']
                    )

    start = timer()

    es = EarlyStopping(monitor='valid_loss',patience=250,min_delta=1e-23)
    model.fit(epochs=1000,batch_size=best_config['batch_size'],X_train=X_train_lvo,y_train=y_train_lvo,X_valid=X_valid_lvo,y_valid=y_valid_lvo,es=best_config['es'])

    end = timer()
    time = end - start

    models.append(model)
    times.append(time)
    
    model.plot_metrics(show=False,save_path='NeuralNetwork/cup/plots/final_retraining_{}.png'.format(trial+1))




losses = [model.history['valid_loss'][-1] for model in models]
best_loss = np.argmin(losses)

print("Losses: {}".format(losses),
        "Best_index: {}".format(best_loss),
        "Best_loss: {}".format(losses[best_loss]),
        "Time: {}".format(times[best_loss]),sep='\n')

best_model = models[best_loss]

write = False
if not os.path.isfile("NeuralNetwork/cup/model.pickle") or write:
    with open('NeuralNetwork/cup/model.pickle','wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

load = False
if load and os.path.isfile("NeuralNetwork/cup/model.pickle"):    
    with open('NeuralNetwork/cup/model.pickle','rb') as handle:
        best_model = pickle.load(handle)


results = best_model.evaluate(X_test,y_test,metrics=['mee'])
print(results)



