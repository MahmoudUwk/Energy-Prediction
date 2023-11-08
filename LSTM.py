# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:27:19 2023

@author: mahmoud
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import  LSTM, BatchNormalization,Dense#,Bidirectional
from keras.models import  Sequential #,load_model
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results
# from sklearn.datasets import fetch_openml
columns=['Steps', 'LSTM Units', 'RMSE','NRMSE', 'Best Epoch', 'Num epochs','seq_length']

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
save_path = 'C:/Users/msallam/Desktop/Kuljeet/results'
option = 2
X_train,y_train,X_test,y_test = get_SAMFOR_data(option)
print(X_train.shape,X_test.shape)
X_train = expand_dims(X_train)
y_train = expand_dims(y_train)
X_test = expand_dims(X_test)
# y_test = expand_dims(y_test)
#%%
#%% LSTM model
#units = 5
adam=Adam(learning_rate=1e-3)
rmspr = RMSprop()
opt_chosen = rmspr
epochs_num = 400
drop_out = 0
#model_name = "CPU_WLP_TF"
#filepath = 'C:/Users/mahmo/OneDrive/Desktop/IS-Wireless/Code/paper_models/models/'+model_name

def get_LSTM_model(units,input_dim,output_dim):
    
    model = Sequential()
    model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True,dropout = drop_out))
    # model.add(BatchNormalization())
    model.add(LSTM(units=units,return_sequences = True,dropout = drop_out))
    model.add(LSTM(units=units,return_sequences = False,dropout = drop_out))

    #model.add(LSTM(units=output_dim,return_sequences = False,dropout = drop_out))
    model.add(Dense(output_dim))
    return model
#model.add(Dense(y_test.shape[-1]))
units = 20
input_dim=(X_train.shape[1],X_train.shape[2])
output_dim = y_test.shape[-1]
model = get_LSTM_model(units,input_dim,output_dim)
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = SaveBestModel()
callbacks_list = [checkpoint]

model.compile(optimizer=opt_chosen, loss='mse')
# model.summary()
# ,callbacks=callbacks_list
history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=64, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
# model.set_weights(checkpoint.best_weights)
# model.save(filepath)
best_epoch = np.argmin(history.history['val_loss'])
y_test_pred = model.predict(X_test)
#%%
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)

alg_name = 'LSTM'
row = [alg_name,rmse,mae,mape]
log_results(row)
#%%
plt.figure(figsize=(10,5))
plt.plot(np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
plt.savefig(os.path.join(save_path,'LSTM.png'))