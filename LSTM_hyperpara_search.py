# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:02:32 2023

@author: mahmo
"""

import matplotlib.pyplot as plt
import os
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, cross_val_score
from keras.layers import Dense,LSTM,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM
import tensorflow as tf
from preprocess_data import save_object,RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM,log_results_HOME_C,inverse_transf
from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import numpy as np
from niapy.algorithms.modified import Mod_FireflyAlgorithm
from niapy.algorithms.basic import FireflyAlgorithm

from niapy.algorithms.basic import BeesAlgorithm


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
                
def get_hyperparameters(x):
    """Get hyperparameters for solution `x`."""
    units = int(x[0]*40 + 8)
    num_layers = int(x[1]*2)+1
    seq = int(x[2]*23 + 6)
    lr = x[3]*2e-2 + 0.5e-3
    params =  {
        'units': units,
        'num_layers': num_layers,
        'seq':seq,
        'lr':lr
    }
    print(params)
    return params

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

def get_LSTM_model(input_dim,output_dim,units,num_layers, seq,lr,name='LSTM_HP'):
    model = Sequential(name=name)
    flag_seq = True
    if num_layers == 1:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False))
    else:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True))
    for dummy in range(num_layers-1):
        if dummy == num_layers-1:
            flag_seq = False     
        model.add(LSTM(units=units,return_sequences = flag_seq))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def get_classifier(x,input_dim,output_dim):
    """Get classifier from solution `x`."""
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    return get_LSTM_model(input_dim,output_dim,**params)

def get_data(x,option,datatype_opt):
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    X_train,y_train,X_test,y_test,save_path,test_time_axis,scaler = get_SAMFOR_data(option,datatype_opt ,params['seq'])
    return X_train,y_train,X_test,y_test,save_path,test_time_axis,scaler

class LSTMHyperparameterOptimization(Problem):
    def __init__(self, option,datatype_opt,num_epoc):
        super().__init__(dimension=4, lower=0, upper=1)
        self.option = option
        self.datatype_opt = datatype_opt
        self.num_epoc = num_epoc

    def _evaluate(self, x):
        X_train,y_train,X_test,y_test,save_path,test_time,scaler = get_data(x,self.option,self.datatype_opt)
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        y_train = expand_dims(expand_dims(y_train))
        y_test = expand_dims(expand_dims(y_test))
        model = get_classifier(x,input_dim,output_dim)
        print(X_train.shape,y_train.shape)
        checkpoint = SaveBestModel()
        callback_es = EarlyStopping(monitor='val_loss', patience=20)
        callbacks_list = [checkpoint,callback_es]
        model.fit(X_train, y_train, epochs=self.num_epoc , batch_size=2**10, verbose=0, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
        model.set_weights(checkpoint.best_weights)
        # hp = get_hyperparameters(x)
        # mse = model.evaluate(X_test,y_test)
        # row = ['FF_LSTM_search',mse,mse,mse,hp['seq'],hp['num_layers'],hp['units'],0,self.datatype_opt,0,0]
        # log_results_LSTM(row,self.datatype_opt,save_path)
        return  model.evaluate(X_test,y_test)

option = 3
datatype_opts = ['5T'] #['1s','1T','15T','30T','home','1s']
run_search= 1
num_epoc = 2500
for datatype_opt in datatype_opts:
    #%%
    if run_search: 
        problem = LSTMHyperparameterOptimization(option,datatype_opt,num_epoc)
        task = Task(problem, max_iters=20, optimization_type=OptimizationType.MINIMIZATION)
        # algorithm = FireflyAlgorithm(population_size = 10)
        algorithm = Mod_FireflyAlgorithm.Mod_FireflyAlgorithm(population_size = 10)
        
        best_params, best_mse = algorithm.run(task)
        
        print('Best parameters:', get_hyperparameters(best_params))
        
        _,_,_,_,save_path,_,_ = get_data(best_params,option,datatype_opt)
        task.plot_convergence(x_axis='evals')
        # a,b = task.convergence_data(x_axis='evals')
        plt.savefig(os.path.join(save_path,'Conv_FF_eval'+str(datatype_opt)+'.png'))
        plt.close()
        
        task.plot_convergence()
        # a,b = task.convergence_data()
        plt.savefig(os.path.join(save_path,'Conv_FF_itr'+str(datatype_opt)+'.png'))
        plt.close()


    #%%
    
    train_option = 1

    
    
    if train_option or run_search:
        params = get_hyperparameters(best_params)

        X_train,y_train,X_test,y_test,save_path,test_time_axis,scaler = get_data(best_params,option,datatype_opt)

        y_train = expand_dims(expand_dims(y_train))
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        
        model = get_classifier(best_params,input_dim,output_dim)
        
        checkpoint = SaveBestModel()
        callbacks_list = [checkpoint]
        
        history = model.fit(X_train, y_train, epochs=num_epoc, batch_size=2**9, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
        
        model.set_weights(checkpoint.best_weights)
        model.save('LSTMFF'+str(datatype_opt))
        best_epoch = np.argmin(history.history['val_loss'])
    ##%%
    else:
        import keras
        params = {'units': 34, 'num_layers': 0, 'seq': 5, 'lr': 0.0026105819848050325}
        _,_,X_test,y_test,save_path,test_time_axis,scaler = get_data(params,option,datatype_opt)
        
        filepath = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy-Prediction/LSTMFF0'
        model = keras.models.load_model(filepath)
        best_epoch = 0#np.argmin(history.history['val_loss'])
        
    y_test = inverse_transf(y_test,scaler)   
    y_test_pred = inverse_transf(model.predict(X_test),scaler)
    rmse = RMSE(y_test,y_test_pred)
    mae = MAE(y_test,y_test_pred)
    mape = MAPE(y_test,y_test_pred)
    print(rmse,mae,mape)
    alg_name = algorithm.Name[0]
    
#%%
    row = [alg_name,rmse,mae,mape,params['seq'],params['num_layers']+1,params['units'],best_epoch,datatype_opt,0,0]
    if datatype_opt == 4:
        log_results_HOME_C(row,datatype_opt,save_path)
    else:
        log_results_LSTM(row,datatype_opt,save_path)
    
    plt.figure(figsize=(20,7),dpi=120)
    plt.plot(test_time_axis,1000*np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(test_time_axis,1000*np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.xticks( rotation=25 )
    plt.ylabel('mW')
    plt.title('Energy Prediction using '+alg_name)
    plt.show()
    info_loop = [params['seq'],params['num_layers']+1,params['units'],best_epoch,datatype_opt]
    name_sav = ""
    for n in info_loop:
        name_sav = name_sav+str(n)+"_" 
    plt.savefig(os.path.join(save_path,'LSTM'+name_sav+'.png'))
    plt.close()
    filename = os.path.join(save_path,alg_name+'.obj')
    obj = {'y_test':y_test,'y_test_pred':y_test_pred}
    save_object(obj, filename)

