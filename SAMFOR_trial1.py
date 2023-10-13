# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:04:56 2023

@author: mahmo
"""

from preprocess_data import scaling_input,slice_data,RMSE,MAE,MAPE,sliding_windows
from lssvr import LSSVR
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from preprocess_data import preprocess,RMSE,MAE,MAPE


# data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz/1477227096132.csv"
# save_path = "C:/Users/msallam/Desktop/Kuljeet/results"
data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv"
save_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/results"
df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
#%%
seq_length = 6
k_step = 1
df = np.array(df["P"])
df = np.array(df[:int(len(df)*0.1)])
train_per = 0.8
len_data = df.shape[0]
train_len = int(train_per*len_data)
train_len_SARIMA = 3600*5#int(SARIMA_per*train_len)
train_len_LSSVR = train_len-train_len_SARIMA
test_len = len_data - train_len
#%%
a = df[:train_len].min()
b = df[:train_len].max()
df_normalized = np.array(scaling_input(df,a,b))
# train_SARIMA = df_normalized[:train_len_SARIMA]
SARIMA_linear_pred = np.array(pd.read_csv(os.path.join(save_path,'SARIMA_linear_prediction.csv')))
train_LSSVR = df_normalized[train_len_SARIMA:train_len_SARIMA+train_len_LSSVR]
# train_LSSVR = np.concatenate((train_LSSVR[seq_length:train_len_LSSVR,np.newaxis],SARIMA_linear_pred[:train_len_LSSVR-seq_length]),axis=0)
testset = df_normalized[train_len:]
print(train_LSSVR.shape,testset.shape)

#%%
del df,df_normalized
X_LSSVR ,y_LSSVR  = sliding_windows(train_LSSVR, seq_length, k_step)
X_LSSVR = np.concatenate((X_LSSVR,SARIMA_linear_pred[:train_len_LSSVR-seq_length]),axis=1)
del train_LSSVR
#%%
clf = LSSVR('C=100','gamma=0.1')
clf.fit(X_LSSVR, np.squeeze(y_LSSVR))
del X_LSSVR,y_LSSVR
#%%
X_test ,y_test  = sliding_windows(testset, seq_length, k_step)
X_test = np.concatenate((X_test,SARIMA_linear_pred[train_len_LSSVR-seq_length:]),axis=1)
del testset
y_test_pred = clf.predict(X_test).reshape(-1,1)

rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print('rmse:',rmse,'||mape:',mape,'||mae:',mae)
#%%
plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
plt.close()


