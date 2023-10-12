# from preprocess_data import preprocess,RMSE,MAE,MAPE
# from pmdarima import auto_arima
import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz/1477227096132.csv"
# save_path = "C:/Users/msallam/Desktop/Kuljeet/results"
data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv"
save_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/results"
percentage_train = 0.7
seq_length = 10
k_step = 1
slide_or_slice = 1 #0 for slide and  1 for slice
# X_train,y_train,X_test,y_test,const_max,const_min = preprocess(data_path,percentage_train,seq_length,k_step,slide_or_slice)

df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
df = df["P"][:100000]
# train_len = int(0.7*len(df))
test_len = 10*3600
train_len = len(df)-test_len
#%%
from pmdarima.model_selection import train_test_split
train, test = train_test_split(df, train_size=train_len)

model = pm.auto_arima(train, p=1,d=0,q=1,P=48,seasonal=True, m=24,stepwise=True,maxiter=10)
print(test.shape[0])
forecasts = model.predict(test.shape[0])

x = np.arange(df.shape[0])
plt.plot(x[:train_len], train, c='blue')
plt.plot(x[train_len:], forecasts, c='green')
plt.show()

# stepwise_model = auto_arima(train, p=1, q=1 , m=48,
                           # P=48, seasonal=True,
                           # d=0, D=0,Q=48, trace=True,
                           # error_action='ignore',  
                           # suppress_warnings=True, 
                           # stepwise=True)
