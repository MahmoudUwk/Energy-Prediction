from sklearn.svm import SVR
import pandas as pd
import numpy as np
# from sklearn.datasets import fetch_openml




def slice_data(data, seq_length,k_step):
    if (len(data)%(seq_length+k_step))!= 0: 
        rem = len(data)%(seq_length+k_step)
        data = data[:-rem]
    data_sliced = np.array(data).reshape(-1,seq_length+k_step)
    return data_sliced[:,:seq_length],np.squeeze(data_sliced[:,seq_length:seq_length+k_step])

df = pd.read_csv("C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv")
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)

training_size = int(len(df) * 0.7)

def RMSE(test,pred):
    return np.sqrt(np.mean((test - pred)**2))

def MAE(test,pred):
    return np.mean(np.abs(pred - test))

def MAPE(test,pred):
    return np.mean(np.abs(pred - test)/np.abs(test))

def scaling_input(X,a,b):
    return (X - a) / (b-a)
#%%
seq_length = 64
k_step = 1
target_col = "P"
X_train , y_train= slice_data(df[target_col][:training_size], seq_length,k_step)
X_test , y_test= slice_data(df[target_col][training_size:], seq_length,k_step)
const_max = X_train.max()
const_min = X_train.min()
X_train = scaling_input(X_train,const_min,const_max)
y_train = scaling_input(y_train,const_min,const_max)
X_test = scaling_input(X_test,const_min,const_max)
y_test = scaling_input(y_test,const_min,const_max)
#%%
# X_train , y_train= sliding_windows(df[target_col][:training_size], seq_length,k_step)
# X_test , y_test= sliding_windows(df[target_col][training_size:], seq_length,k_step)
del df

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



