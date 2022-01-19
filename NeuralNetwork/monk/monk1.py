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
                'n_units':3,
                'batch_size':batch,
                'out_units':1,
                'hidden_act':'sigmoid',
                'out_act':'sigmoid',
                'weights_init':'xavier_uniform',
                'lr':0.83,
                'momentum':0.9,
                'reg_type':None,
                'lambda':3e-4,
                'lr_decay':False,
                'nesterov':False
            }

#final model training and assessment
start = timer()

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

es = EarlyStopping(monitor='valid_loss',patience=10,min_delta=1e-23)
model.fit(epochs=1000,batch_size=config['batch_size'],X_train=X_train,y_train=y_train,X_valid=X_test,y_valid=y_test,es=es)

end = timer()
time_it = end-start
print("\nTime elapsed:\t{}".format(time_it), end='\r')
print("Best config:\t",config)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,7))
ax1.plot(model.history['train_loss'], label='train_loss')
ax1.plot(model.history['valid_loss'], color='tab:orange', linestyle='dashed',label='valid_loss')
ax2.plot(model.history['train_accuracy'], label='train_accuracy')
ax2.plot(model.history['valid_accuracy'], color='tab:orange', linestyle='dashed',label='valid_accuracy')
plt.legend()
plt.grid()
plt.show()