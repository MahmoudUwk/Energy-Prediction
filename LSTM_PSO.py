
# import time
import numpy as np
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM
# from IPython.display import Image
# from keras.callbacks import TensorBoard
from keras.layers import Dense,LSTM,Flatten
from keras.models import Sequential
from pyswarms.single.global_best import GlobalBestPSO
# from sklearn.datasets import load_iris,load_diabetes,fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.optimizers import Adam

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()
                
def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))
#%%
option = 3
datatype_opt = 0
seq = 6
X_train,y_train,X_test,y_test,save_path = get_SAMFOR_data(option,datatype_opt,seq)

# Data set transformation
#%%



y_train = expand_dims(expand_dims(y_train))
if len(X_train.shape)<3:
    X_train = expand_dims(X_train)
    
    X_test = expand_dims(X_test)
# diabetes  = load_diabetes()
# X, y = diabetes.data, diabetes.target
# X, y = fetch_california_housing(return_X_y=True, as_frame=True)
# X = iris['data']
# y = iris['target']
# names = iris['target_names']
# feature_names = iris['feature_names']
# enc = OneHotEncoder()
# Y = enc.fit_transform(y[:, np.newaxis]).toarray()
# scaler = MinMaxScaler()#StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = np.expand_dims(X_scaled, axis = 1)
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_scaled, y, test_size=0.5, random_state=2)
# n_features = X.shape[1]
# # n_classes = y.shape[1]
# n_classes = 1

## Building the neural network


# def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
#     model = Sequential(name=name)
#     for i in range(n):
#         model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
#     model.add(Dense(output_dim, activation='softmax'))
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model

def get_LSTM_model(input_dim,output_dim,units,num_layers, name='model_LSTM_PSO'):
    model = Sequential(name=name)
    model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True))
    model.add(Flatten())
    model.add(Dense(output_dim))
    # model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    # model.compile(optimizer=adam, loss='mse')
    return model
#%%
n_layers = 0
units = 8
input_dim=(X_train.shape[1],X_train.shape[2])
output_dim = y_train.shape[-1]
model = get_LSTM_model(input_dim, output_dim,units, n_layers)
adam=Adam(learning_rate=2e-3)
epochs_num = 2500
checkpoint = SaveBestModel()
callbacks_list = [checkpoint]

model.compile(optimizer=adam, loss='mse')
# model.summary()
# ,callbacks=callbacks_list
history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=2048, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
model.set_weights(checkpoint.best_weights)
# model.save(filepath)
best_epoch = np.argmin(history.history['val_loss'])
y_test_pred = model.predict(X_test)
#%%
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)

alg_name = 'lstm'
row = [alg_name,rmse,mae,mape,seq,n_layers+2,units,best_epoch,datatype_opt]
log_results_LSTM(row,datatype_opt,save_path)

#%%
## Building the PSO function and optimization
def get_shape(model):
    weights_layer = model.get_weights()
    shapes = []
    dimension = 0
    for weights in weights_layer:
        shapes.append(weights.shape)
        dimension = dimension + np.prod(weights.shape)
    return shapes,dimension
def set_shape(weights,shapes):
    new_weights = []
    index=0
    for shape in shapes:
        if(len(shape)>1):
            n_nodes = np.prod(shape)+index
        else:
            n_nodes=shape[0]+index
        tmp = np.array(weights[index:n_nodes]).reshape(shape)
        new_weights.append(tmp)
        index=n_nodes
    return new_weights

#%%

def evaluate_nn(W, shape,X, Y):
    result = np.zeros((len(W)))
    for counter,weights in enumerate(W):
        model.set_weights(set_shape(weights,shape))
        score = model.evaluate(X, Y, verbose=0)
        result[counter]= score
    return result

shape,dimension = get_shape(model)
print('Dimension-----',dimension)
x_max = 1.0 * 15*np.ones(dimension)
x_min = -1.0 * x_max
bounds = (x_min, x_max)
options = {'c1': 0.3, 'c2': 0.8, 'w': 0.4}
optimizer = GlobalBestPSO(n_particles=100, dimensions=dimension,
                          options=options, bounds=bounds)
cost, pos = optimizer.optimize(evaluate_nn, 40, n_processes=None, verbose=True, X=X_train, Y=y_train,shape=shape)
model.set_weights(set_shape(pos,shape))
#%%
pred = model.predict(X_test)
score = RMSE(pred,y_test)
print('RMSE:', score)

#%% fine tuning
from keras.optimizers import Adam
adam=Adam(learning_rate=2e-3)
model.set_weights(set_shape(pos,shape))
model.compile(optimizer=adam, loss='mse')
epochs_tune = 1000
history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=2048, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
model.set_weights(checkpoint.best_weights)

best_epoch = np.argmin(history.history['val_loss'])
y_test_pred = model.predict(X_test)

rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)

alg_name = 'LSTM_PSO_fined_tuned'
row = [alg_name,rmse,mae,mape,seq,n_layers+2,units,best_epoch,datatype_opt]
log_results_LSTM(row,datatype_opt,save_path)
