from sklearn.svm import SVR
import pandas as pd
import numpy as np
# from sklearn.datasets import fetch_openml
import os

from matplotlib import pyplot as plt
# from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from preprocess_data import*
import time

option = 2
datatype_opt= '5T'
seq = 7 
X_train,y_train,X_test,y_test,save_path,test_time_axis,scaler = get_SAMFOR_data(option,datatype_opt,seq)

y_test = inverse_transf(y_test,scaler)

print(X_train.shape,X_test.shape)
#%%

clf = RandomForestRegressor(random_state=0)

print('start training')
start_train = time.time()
clf.fit(X_train, y_train)
end_train = time.time()
print('End training')
train_time = (end_train - start_train)/60

start_test = time.time()
y_test_pred = inverse_transf(clf.predict(X_test),scaler)
end_test = time.time()
test_time = end_test - start_test



alg_name = 'RFR'
name_sav = os.path.join(save_path,'RFR_datatype_opt'+str(datatype_opt)+'.png')
plot_test(test_time_axis,y_test,y_test_pred,name_sav,alg_name)


rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)

filename = os.path.join(save_path,alg_name+'.obj')
obj = {'y_test':y_test,'y_test_pred':y_test_pred}
save_object(obj, filename)
#%%

# row = [alg_name,rmse,mae,mape,seq,0,0,0,datatype_opt,train_time,test_time]
# log_results_HOME_C(row,datatype_opt,save_path)

row = [alg_name,rmse,mae,mape,seq,train_time,test_time]
log_results(row,datatype_opt,save_path)
#%%

new_clf = SVR(C=10, epsilon=0.01,kernel='rbf')
start_train = time.time()
new_clf.fit(X_train, y_train)
end_train = time.time()
train_time = (end_train - start_train)/60

start_test = time.time()
y_test_pred = inverse_transf(new_clf.predict(X_test),scaler)
end_test = time.time()
test_time = end_test - start_test


alg_name = 'SVR'
name_sav = os.path.join(save_path,'SVR_datatype_opt'+str(datatype_opt)+'.png')
plot_test(test_time_axis,y_test,y_test_pred,name_sav,alg_name)

rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)


if datatype_opt == 4:
    row = [alg_name,rmse,mae,mape,seq,0,0,0,datatype_opt,train_time,test_time]
    log_results_HOME_C(row,datatype_opt,save_path)
else:
    row = [alg_name,rmse,mae,mape,seq,train_time,test_time]
    log_results(row,datatype_opt,save_path)

filename = os.path.join(save_path,alg_name+'.obj')
obj = {'y_test':y_test,'y_test_pred':y_test_pred}
save_object(obj, filename)
#%%

# new_clf =  MLPRegressor(max_iter=500)
# new_clf.fit(X_train, y_train)

# y_test_pred = new_clf.predict(X_test)

# plt.figure(figsize=(10,5))
# plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
# plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
# plt.legend(['Actual','Predicted'])
# plt.xlabel('Timestamp')
# plt.show()
# plt.savefig(os.path.join(save_path,'MLP_sklearn.png'))
# # print('SVR RMSE:',mean_squared_error(y_test,y_test_pred))
# rmse = RMSE(y_test,y_test_pred)
# mae = MAE(y_test,y_test_pred)
# mape = MAPE(y_test,y_test_pred)
# print(rmse,mae,mape)

# alg_name = 'MLP sklearn'
# if datatype_opt == 4:
#     row = [alg_name,rmse,mae,mape,seq,0,0,0,datatype_opt,train_time,test_time]
#     log_results_HOME_C(row,datatype_opt,save_path)
# else:
#     row = [alg_name,rmse,mae,mape,seq,train_time,test_time]
#     log_results(row,datatype_opt,save_path)