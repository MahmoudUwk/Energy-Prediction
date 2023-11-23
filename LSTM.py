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
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM
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
def get_LSTM_model(units,input_dim,output_dim,num_layers):
    
    model = Sequential()
    model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True,dropout = drop_out))
    # model.add(BatchNormalization())
    for dummy in range(num_layers):
        model.add(LSTM(units=units,return_sequences = True,dropout = drop_out))
    model.add(LSTM(units=units,return_sequences = False,dropout = drop_out))  
    model.add(Dense(output_dim))
    return model
#%%
save_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models'#'C:/Users/msallam/Desktop/Energy Prediction/results'
option = 3
alg_name = 'LSTM'
data_types = [0,1,2]
num_layers_all = [1,2]
num_units = [8,12,16,32,64]
seq = [6,8,10]
for datatype_opt in data_types:
    for seq_length in seq:
        X_train,y_train,X_test,y_test = get_SAMFOR_data(option,datatype_opt,seq_length)
        seq = X_train.shape[1]
        
        # X_train = expand_dims(X_train)
        y_train = expand_dims(expand_dims(y_train))
        # X_test = expand_dims(X_test)
        # y_test = expand_dims(y_test)
        #%%
        #%% LSTM model
        #units = 5
        adam=Adam()#learning_rate=1e-3)
        rmspr = RMSprop()
        opt_chosen = rmspr
        epochs_num = 400
        drop_out = 0
    
    
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        print(input_dim,output_dim)
        for num_layers in num_layers_all:
            for units in num_units:
                model = get_LSTM_model(units,input_dim,output_dim,num_layers)
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
                rmse = RMSE(y_test,y_test_pred)
                mae = MAE(y_test,y_test_pred)
                mape = MAPE(y_test,y_test_pred)
                print(rmse,mae,mape)
                
                
                row = [alg_name,rmse,mae,mape,seq,num_layers+2,units,best_epoch,datatype_opt]
                log_results_LSTM(row)
                #%%
                plt.figure(figsize=(10,5))
                plt.plot(np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
                plt.plot(np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
                plt.legend(['Actual','Predicted'])
                plt.xlabel('Timestamp')
                plt.show()
                plt.savefig(os.path.join(save_path,'LSTM'+str(datatype_opt)+'.png'))