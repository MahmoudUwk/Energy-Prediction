from sklearn.svm import SVR
import pandas as pd
import numpy as np
# from sklearn.datasets import fetch_openml
import os
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results

# save_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models'
save_path = 'C:/Users/msallam/Desktop/Kuljeet/results'
option = 2
X_train,y_train,X_test,y_test = get_SAMFOR_data(option)
print(X_train.shape,X_test.shape)
#%%
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
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
row = [alg_name,rmse,mae,mape]
log_results(row)
#%%
from sklearn.ensemble import RandomForestRegressor
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
row = [alg_name,rmse,mae,mape]
log_results(row)
