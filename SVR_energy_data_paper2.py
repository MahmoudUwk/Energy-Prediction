from sklearn.svm import SVR
import pandas as pd
import numpy as np
# from sklearn.datasets import fetch_openml
from preprocess_data import get_SAMFOR_data
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
percentage_data_use = 0.15
k_step = 1
percentage_train = 0.8
SARIMA_len = 3600
option = 2
SARIMA_pred = os.path.join(save_path, 'SARIMA_linear_prediction.csv')
X_train,y_train,X_test,y_test = get_SAMFOR_data(df,seq_length,k_step,percentage_data_use,percentage_train,SARIMA_len,option,SARIMA_pred)
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
# print('SVR RMSE:',mean_squared_error(y_test,y_test_pred))
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)
#%%
from sklearn.ensemble import RandomForestRegressor

# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
# print('RF RMSE:',mean_squared_error(y_test,y_test_pred))
rmse = RMSE(y_test,y_test_pred)
mae = MAE(y_test,y_test_pred)
mape = MAPE(y_test,y_test_pred)
print(rmse,mae,mape)



