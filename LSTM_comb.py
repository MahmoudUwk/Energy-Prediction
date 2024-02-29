# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:01:18 2024

@author: mahmo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:27:19 2023

@author: mahmoud
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from keras.layers import  Bidirectional,LSTM, BatchNormalization,Dense#,Bidirectional
from keras.models import  Sequential #,load_model
from keras.optimizers import Adam,RMSprop,Adadelta
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import keras
from preprocess_data import *#RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM,log_results_HOME_C,inverse_transf
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
# def get_LSTM_model(units,input_dim,output_dim,num_layers):
#     model = Sequential()
#     if num_layers == 1:
#         model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False))
#     else:
#         model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True))
#     for dummy in range(num_layers-1):
#         model.add(LSTM(units=units,return_sequences = False))
#     model.add(Dense(output_dim))
#     return model

def get_LSTM_model(units,input_dim,output_dim,num_layers,name='LSTM_HP'):
    model = Sequential(name=name)
    flag_seq = True
    if num_layers == 1:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False))
    else:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True))
    for dummy in range(num_layers-1):
        if dummy == num_layers-2:
            flag_seq = False     
        model.add(LSTM(units=units,return_sequences = flag_seq))
    model.add(Dense(output_dim))
    return model
#%%
option = 3
alg_name = 'LSTM'
data_types = ['5T']
seq_all = [7]#[5,20]
num_units = [10]#[35]#[8,10,15]
num_layers_all = [1]

# lr = 0.0026105819848050325
# lr = 0.0005
lr = 0.001
adam=Adam(learning_rate=lr)
# rmspr = RMSprop()
opt_chosen = adam
        
for datatype_opt in data_types:
    for seq_counter , seq in enumerate(seq_all):
        X_train,y_train,X_test,y_test,save_path,test_time_axis,scaler = get_SAMFOR_data(option,datatype_opt,seq)
        y_test = inverse_transf(y_test,scaler)
        # X_train = X_train[:,:,0]
        # X_test = X_test[:,:,0]
        print(X_train.shape)
        y_train = expand_dims(expand_dims(y_train))
        if len(X_train.shape)<3:
            X_train = expand_dims(X_train)
            
            X_test = expand_dims(X_test)
        print(X_train.shape,y_train.shape,X_test.shape)
        # y_test = expand_dims(y_test)
        #%%
        #%% LSTM model

        epochs_num = 2000
        drop_out = 0
        callback_falg = 1
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        batch_size_n = 2**12
        units = num_units[seq_counter]
        print(input_dim,output_dim)
        for num_layers in num_layers_all:    
            model = get_LSTM_model(units,input_dim,output_dim,num_layers)            
            model.compile(optimizer=opt_chosen, loss='mse')
            # model.summary()
            # ,callbacks=callbacks_list
            print('start training')
            start_train = time.time()
            if callback_falg:
                checkpoint = SaveBestModel()
                callbacks_list = [checkpoint]
                history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
            else:
                history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=1, shuffle=True)#, validation_split=0.2,callbacks=callbacks_list)
            #%%
            end_train = time.time()
            print('End training')
            train_time = (end_train - start_train)/60
            if callback_falg:
                model.set_weights(checkpoint.best_weights)
                best_epoch =np.argmin(history.history['val_loss'])
            else:
                best_epoch = 0
            # model.save(filepath)
            #%%
            
            start_test = time.time()
            y_test_pred = inverse_transf(model.predict(X_test),scaler)
            end_test = time.time()
            test_time = end_test - start_test
            #%%
            rmse = RMSE(y_test,y_test_pred)
            mae = MAE(y_test,y_test_pred)
            mape = MAPE(y_test,y_test_pred)
            print(rmse,mae,mape)
            # best_epoch = epochs_num
            
            row = [alg_name,rmse,mae,mape,seq,num_layers,units,best_epoch,datatype_opt,train_time,test_time]
            if datatype_opt == 4:
                log_results_HOME_C(row,datatype_opt,save_path)
            else:
                log_results_LSTM(row,datatype_opt,save_path)
            #%%
            plt.figure(figsize=(10,7),dpi=180)
            plt.plot(test_time_axis,1000*np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
            plt.plot(test_time_axis,1000*np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
            plt.legend(['Actual','Predicted'])
            plt.xlabel('Timestamp')
            plt.xticks( rotation=25)
            plt.ylabel('mW')
            plt.title('Energy Prediction using '+alg_name)
            plt.show()
            info_loop = [seq,num_layers,units,best_epoch,datatype_opt]
            name_sav = ""
            for n in info_loop:
                name_sav = name_sav+str(n)+"_" 
            plt.savefig(os.path.join(save_path,'LSTM'+name_sav+'.png'))
            plt.close()
            filename = os.path.join(save_path,alg_name+'.obj')
            obj = {'y_test':y_test,'y_test_pred':y_test_pred}
            save_object(obj, filename)
