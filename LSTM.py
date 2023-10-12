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


def slice_data(data, seq_length,k_step):
    if (len(data)%(seq_length+k_step))!= 0: 
        rem = len(data)%(seq_length+k_step)
        data = data[:-rem]
    data_sliced = np.array(data).reshape(-1,seq_length+k_step)
    return data_sliced[:,:seq_length],np.squeeze(data_sliced[:,seq_length:seq_length+k_step])
path = 'C:/Users/msallam/Desktop/Kuljeet/1Hz'
# path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2'
df = pd.read_csv(path+"/1477227096132.csv")
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)

training_size = int(len(df) * 0.7)

def scaling_input(X,a,b):
    return (X - a) / (b-a)

def RMSE(test,pred):
    return np.sqrt(np.mean((test - pred)**2))

def MAE(test,pred):
    return np.mean(np.abs(pred - test))

def MAPE(test,pred):
    return np.mean(np.abs(pred - test)/np.abs(test))

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))
#%%
seq_length = 64
k_step = 1
target_col = "P"
X_train , y_train= slice_data(df[target_col][:training_size], seq_length,k_step)
X_test , y_test= slice_data(df[target_col][training_size:], seq_length,k_step)
const_max = X_train.max()
const_min = X_train.min()
X_train = expand_dims(scaling_input(X_train,const_min,const_max))
y_train = expand_dims(scaling_input(y_train,const_min,const_max))
X_test = expand_dims(scaling_input(X_test,const_min,const_max))
y_test = expand_dims(scaling_input(y_test,const_min,const_max))
#%%
# X_train , y_train= sliding_windows(df[target_col][:training_size], seq_length,k_step)
# X_test , y_test= sliding_windows(df[target_col][training_size:], seq_length,k_step)
del df
#%%
#%% LSTM model
#units = 5
adam=Adam(learning_rate=2e-3)
rmspr = RMSprop()
opt_chosen = adam
epochs_num = 60
drop_out = 0
#model_name = "CPU_WLP_TF"
#filepath = 'C:/Users/mahmo/OneDrive/Desktop/IS-Wireless/Code/paper_models/models/'+model_name

def get_LSTM_model(units,input_dim,output_dim):
    
    model = Sequential()
    model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False,dropout = drop_out))
    #model.add(BatchNormalization())
    #model.add(LSTM(units=units,return_sequences = True,dropout = drop_out))
    #model.add(LSTM(units=units,return_sequences = False,dropout = drop_out))

    #model.add(LSTM(units=output_dim,return_sequences = False,dropout = drop_out))
    model.add(Dense(output_dim))
    return model
#model.add(Dense(y_test.shape[-1]))
units = 10
input_dim=(seq_length,X_train.shape[2])
output_dim = y_test.shape[-1]
model = get_LSTM_model(units,input_dim,output_dim)
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = SaveBestModel()
callbacks_list = [checkpoint]

model.compile(optimizer=opt_chosen, loss='mse')
# model.summary()
# ,callbacks=callbacks_list
history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=64, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
model.set_weights(checkpoint.best_weights)
# model.save(filepath)
best_epoch = np.argmin(history.history['val_loss'])
y_test_pred = model.predict(X_test)

#%%
plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)

print(rmse,mae,mape)

