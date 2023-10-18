# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:04:56 2023

@author: mahmo
"""

from preprocess_data import get_SAMFOR_data
from lssvr import LSSVR
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from preprocess_data import RMSE,MAE,MAPE


# data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz/1477227096132.csv"
# save_path = "C:/Users/msallam/Desktop/Kuljeet/results"
data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv"
save_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/results"
df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
#%%
seq_length = 6
percentage_data_use = 0.2
k_step = 1
percentage_train = 0.8
SARIMA_len = 3600*5
option = 1

X_LSSVR,y_LSSVR,X_test,y_test = get_SAMFOR_data(df,seq_length,k_step,percentage_data_use,percentage_train,SARIMA_len,option,SARIMA_pred_path=save_path)
print(X_LSSVR.shape,X_test.shape)
#%%
clf = LSSVR(C=100,gamma=0.1,kernel='rbf')
clf.fit(X_LSSVR, np.squeeze(y_LSSVR))

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
# plt.close()


