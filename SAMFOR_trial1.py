# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:04:56 2023

@author: mahmo
"""
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from preprocess_data import*# RMSE,MAE,MAPE,log_results,get_SAMFOR_data,plot_test,plot_test,inverse_transf
from lssvr import LSSVR
from sklearn.svm import LinearSVR
option = 1
datatype_opt = '5T'
seq_length=7
X_LSSVR,y_LSSVR,X_test,y_test,save_path,test_time_axis,scaler = get_SAMFOR_data(option,datatype_opt,seq_length)
print(X_LSSVR.shape,X_test.shape)
y_test = inverse_transf(y_test,scaler)
opt = 1
#%%
if opt ==0:
    alg_name ='SAMFOR_SARIMA_LSSVR'
    # clf = LinearSVR(C=10,epsilon=0.01,max_iter=10000)
    clf = LSSVR(C=1,gamma=0.001,kernel='rbf')
    print('start training')
    start_train = time.time()
    clf.fit(X_LSSVR, np.squeeze(y_LSSVR))
    end_train = time.time()
    print('End training')
    train_time = (end_train - start_train)/60
    
else:
    from sklearn.svm import SVR
    alg_name = 'SAMFOR'
    clf = SVR(C=1, epsilon=0.001,kernel='rbf')
    print('start training')
    start_train = time.time()
    clf.fit(X_LSSVR, np.squeeze(y_LSSVR))
    end_train = time.time()
    print('End training')
    train_time = (end_train - start_train)/60
    
start_test = time.time()
y_test_pred = inverse_transf(np.squeeze(clf.predict(X_test).reshape(-1,1)),scaler)
end_test = time.time()
test_time = end_test - start_test

y_test = np.squeeze(y_test)
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print('rmse:',rmse,'||mape:',mape,'||mae:',mae)
#%%
# seq = X_LSSVR.shape[1]-1
row = [alg_name,rmse,mae,mape,seq_length,train_time,test_time]
log_results(row,datatype_opt,save_path)
#%%
name_sav = os.path.join(save_path,'SAMFOR_datatype_opt'+str(datatype_opt)+'.png')
plot_test(test_time_axis,y_test,y_test_pred,name_sav,alg_name)

filename = os.path.join(save_path,alg_name+'.obj')
obj = {'y_test':y_test,'y_test_pred':y_test_pred}
save_object(obj, filename)