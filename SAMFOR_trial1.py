# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:04:56 2023

@author: mahmo
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from preprocess_data import RMSE,MAE,MAPE,log_results,get_SAMFOR_data
# from lssvr import LSSVR
save_path = 'C:/Users/msallam/Desktop/Kuljeet/results'
option = 1
X_LSSVR,y_LSSVR,X_test,y_test = get_SAMFOR_data(option)
print(X_LSSVR.shape,X_test.shape)
opt = 0
#%%
if opt ==0:
    from sklearn.svm import LinearSVR
    alg_name ='SAMFOR_SARIMA'
    clf = LinearSVR(C=10,epsilon=0.01)
    # clf = LSSVR(C=1,gamma=0.001,kernel='rbf')
    clf.fit(X_LSSVR, np.squeeze(y_LSSVR))
    
else:
    from sklearn.svm import SVR
    alg_name = 'SVR_SAMFOR'
    clf = SVR(C=1, epsilon=0.001,kernel='rbf')
    clf.fit(X_LSSVR, np.squeeze(y_LSSVR))
y_test_pred = clf.predict(X_test).reshape(-1,1)

rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print('rmse:',rmse,'||mape:',mape,'||mae:',mae)
#%%
row = [alg_name,rmse,mae,mape]
log_results(row)
#%%
# save_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models'
plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
plt.savefig(os.path.join(save_path,'SAMFOR.png'))
plt.close()
