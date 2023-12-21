from sklearn.svm import SVR
import pandas as pd
import numpy as np
# from sklearn.datasets import fetch_openml
import os

from matplotlib import pyplot as plt
# from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results,log_results_HOME_C

option = 2
datatype_opt=4
seq=10
X_train,y_train,X_test,y_test,save_path = get_SAMFOR_data(option,datatype_opt,seq)
print(X_train.shape,X_test.shape)
#%%

clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
plt.savefig(os.path.join(save_path,'RFR.png'))
# print('RF RMSE:',mean_squared_error(y_test,y_test_pred))
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)
#%%
alg_name = 'RFR'
if datatype_opt == 4:
    row = [alg_name,rmse,mae,mape,seq,0,0,0,datatype_opt]
    log_results_HOME_C(row,datatype_opt,save_path)
else:
    row = [alg_name,rmse,mae,mape,seq]
    log_results(row,datatype_opt,save_path)
#%%

new_clf = SVR(C=10, epsilon=0.01,kernel='rbf')
new_clf.fit(X_train, y_train)

y_test_pred = new_clf.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
plt.savefig(os.path.join(save_path,'SVR.png'))
# print('SVR RMSE:',mean_squared_error(y_test,y_test_pred))
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)
#%%
alg_name = 'SVR'
if datatype_opt == 4:
    row = [alg_name,rmse,mae,mape,seq,0,0,0,datatype_opt]
    log_results_HOME_C(row,datatype_opt,save_path)
else:
    row = [alg_name,rmse,mae,mape,seq]
    log_results(row,datatype_opt,save_path)

#%%

new_clf =  MLPRegressor(max_iter=500)
new_clf.fit(X_train, y_train)

y_test_pred = new_clf.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
plt.savefig(os.path.join(save_path,'MLP_sklearn.png'))
# print('SVR RMSE:',mean_squared_error(y_test,y_test_pred))
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)

alg_name = 'MLP sklearn'
if datatype_opt == 4:
    row = [alg_name,rmse,mae,mape,seq,0,0,0,datatype_opt]
    log_results_HOME_C(row,datatype_opt,save_path)
else:
    row = [alg_name,rmse,mae,mape,seq]
    log_results(row,datatype_opt,save_path)