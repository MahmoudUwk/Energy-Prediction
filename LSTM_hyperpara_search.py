# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:02:32 2023

@author: mahmo
"""

import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from keras.layers import Dense,LSTM,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM
import tensorflow as tf
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM,log_results_HOME_C
from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import numpy as np
from niapy.algorithms.modified import Mod_FireflyAlgorithm


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
    units = int(x[0]*28 + 4)
    num_layers = int(x[1]*2)
    seq = int(x[2]*12 + 4)
    lr = x[3]*1e-2 + 1e-3
    params =  {
        'units': units,
        'num_layers': num_layers,
        'seq':seq,
        'lr':lr
    }
    return params

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

def get_LSTM_model(input_dim,output_dim,units,num_layers, seq,lr,name='LSTM_HP'):
    model = Sequential(name=name)
    model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True))
    for dummy in range(num_layers):
        model.add(LSTM(units=units,return_sequences = True))
    model.add(Flatten())
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
    X_train,y_train,X_test,y_test,save_path = get_SAMFOR_data(option,datatype_opt ,params['seq'])
    return X_train,y_train,X_test,y_test,save_path

class LSTMHyperparameterOptimization(Problem):
    def __init__(self, option,datatype_opt,num_epoc):
        super().__init__(dimension=4, lower=0, upper=1)
        self.option = option
        self.datatype_opt = datatype_opt
        self.num_epoc = num_epoc

    def _evaluate(self, x):
        X_train,y_train,X_test,y_test,save_path = get_data(x,self.option,self.datatype_opt)
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        y_train = expand_dims(expand_dims(y_train))
        y_test = expand_dims(expand_dims(y_test))
        model = get_classifier(x,input_dim,output_dim)
        checkpoint = SaveBestModel()
        callbacks_list = [checkpoint]
        model.fit(X_train, y_train, epochs=self.num_epoc , batch_size=2**9, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
        model.set_weights(checkpoint.best_weights)
        return model.evaluate(X_test,y_test)

option = 3
datatype_opts = [3,2,1,0]
run_search= 1
num_epoc = 1000
for datatype_opt in datatype_opts:
    #%%
    if run_search:
        problem = LSTMHyperparameterOptimization(option,datatype_opt,num_epoc)
        task = Task(problem, max_evals=20, optimization_type=OptimizationType.MINIMIZATION)
        algorithm = Mod_FireflyAlgorithm.Mod_FireflyAlgorithm(population_size = 20)
        
        
        best_params, best_mse = algorithm.run(task)
        
        print('Best parameters:', get_hyperparameters(best_params))
        
        task.plot_convergence(x_axis='evals')
        a,b = task.convergence_data(x_axis='evals')
        # best_model = get_classifier(best_params)
    
    #extract info, time, convergence change,......,how many iterations,value changes.
    #make a presentation about everything so far, up and downs and current results
    #next week meeting, prepare the presentation by tuesday and show it meeting then
    
    #pick up paper to get idea for another nature inspired algorithm
    #%%
    params = get_hyperparameters(best_params)
    train_option = 1
    # params =  {'units': 15, 'num_layers': 0, 'seq': 7, 'lr': 0.0010540494547447551}
    best_params = params
    
    
    if train_option:
        
        X_train,y_train,X_test,y_test,save_path = get_data(best_params,option,datatype_opt)
        y_train = expand_dims(expand_dims(y_train))
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        
        model = get_classifier(best_params,input_dim,output_dim)
        
        checkpoint = SaveBestModel()
        callbacks_list = [checkpoint]
        
        history = model.fit(X_train, y_train, epochs=num_epoc, batch_size=2**9, verbose=1, shuffle=True, validation_split=0.2,callbacks=callbacks_list)
        
        model.set_weights(checkpoint.best_weights)
        model.save('LSTMFF'+str(datatype_opt))
    ##%%
    else:
        import keras
        _,_,X_test,y_test,save_path = get_data(best_params,option,datatype_opt)
        filepath = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy-Prediction/LSTMFF0'
        model = keras.models.load_model(filepath)
        
        
    y_test_pred = model.predict(X_test)
    rmse = RMSE(y_test,y_test_pred)
    mae = MAE(y_test,y_test_pred)
    mape = MAPE(y_test,y_test_pred)
    print(rmse,mae,mape)
    alg_name = 'Mof_FF_LSTM'
    best_epoch = np.argmin(history.history['val_loss'])
    row = [alg_name,rmse,mae,mape,params['seq'],params['num_layers']+1,params['units'],best_epoch,datatype_opt]
    if datatype_opt == 4:
        log_results_HOME_C(row,datatype_opt,save_path)
    else:
        log_results_LSTM(row,datatype_opt,save_path)
    
    plt.figure(figsize=(10,5))
    plt.plot(np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.show()
    info_loop = [params['seq'],params['num_layers']+1,params['units'],best_epoch,datatype_opt]
    name_sav = ""
    for n in info_loop:
        name_sav = name_sav+str(n)+"_" 
    plt.savefig(os.path.join(save_path,'LSTMFF'+name_sav+'.png'))
    plt.close()

