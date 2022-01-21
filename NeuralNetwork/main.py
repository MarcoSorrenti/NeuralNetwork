import os, pickle
import matplotlib.pyplot as plt
from neuralnetwork.model.Optimizer import EarlyStopping
from neuralnetwork.datasets.util import load_cup, train_test_split
from neuralnetwork.model.NeuralNetwork import build_model
from neuralnetwork.model_selection import GridSearchCVNNParallel
import os

X_train, X_test_blind, y_train, y_test_blind = load_cup()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

X_train_lvo, X_valid_lvo, y_train_lvo, y_valid_lvo = train_test_split(X_train, y_train)

n_features = X_train.shape[1]
batch = len(X_train)

run_grid = False
grid_search_config = False

if not os.path.isfile('NeuralNetwork/best_config.pickle') or run_grid:

    es = EarlyStopping(monitor='valid_loss',patience=250,min_delta=1e-23)

    params_config = {
                'n_features': [n_features],
                'n_hidden_layers':[1, 2],
                'n_units':[20, 40, 80, 100],
                'batch_size':[128, int(batch/2)],
                'out_units':[2],
                'hidden_act':['tanh'],
                'out_act':['linear'],
                'weights_init':['he_uniform'],
                'lr':[0.0001, 0.001, 0.01],
                'momentum':[0.2, 0.5, 0.9],
                'reg_type': ['l2'],
                'lambda':[0.0001, 0.001, 0.01],
                'lr_decay':[True, False],
                'nesterov':[True, False],
                'es':[es]
                }

    if __name__ == '__main__':

        gs = GridSearchCVNNParallel(params_config)
        gs.fit(X_train,y_train,loss='mee',epochs=1000,shuffle=True,n_jobs=8)
        gs_results = sorted(gs.grid_results, key = lambda i: (i['mean_error_valid'], i['st_dev_valid'],i['time']))
        best_configs = [config for config in gs_results[:10]]

    with open('NeuralNetwork/best_config.pickle', 'wb') as handle:
        pickle.dump(best_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)



#chosing grid search best configuration or a custom
if grid_search_config:

    with open('NeuralNetwork/best_config.pickle', 'rb') as handle:
        gs_best_configs = pickle.load(handle)
        
    print(gs_best_configs[:2])

    best_config = gs_best_configs[0]['parameters']

else:

    es = EarlyStopping(monitor='valid_loss',patience=200,min_delta=1e-23)

    best_config = {
                'n_features': n_features,
                'n_hidden_layers':2,
                'n_units':100,
                'batch_size':int(batch/2),
                'out_units':2,
                'hidden_act':'tanh',
                'out_act':'linear',
                'weights_init':'he_normal',
                'lr':0.001,
                'momentum':0.9,
                'reg_type':'l2',
                'lambda':0.00001,
                'lr_decay':False,
                'nesterov':True,
                'es':es
                }


#final model training and assessment
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

es = EarlyStopping(monitor='valid_loss',patience=200,min_delta=1e-23)
model.fit(epochs=1000,batch_size=best_config['batch_size'],X_train=X_train_lvo,y_train=y_train_lvo,X_valid=X_valid_lvo,y_valid=y_valid_lvo,es=es)

print(best_config)

model.plot_metrics()

