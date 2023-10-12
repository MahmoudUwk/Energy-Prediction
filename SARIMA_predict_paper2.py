# from preprocess_data import preprocess,RMSE,MAE,MAPE
# from pmdarima import auto_arima
import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz/1477227096132.csv"
save_path = "C:/Users/msallam/Desktop/Kuljeet/results"
# data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv"
# save_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/results"
percentage_train = 0.7
seq_length = 10
k_step = 1
slide_or_slice = 1 #0 for slide and  1 for slice
# X_train,y_train,X_test,y_test,const_max,const_min = preprocess(data_path,percentage_train,seq_length,k_step,slide_or_slice)

df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
test_len = 60
df = df["P"][:3*3600+test_len]
# train_len = int(0.7*len(df))
train_len = len(df)-test_len

from pmdarima.model_selection import train_test_split
train, test = train_test_split(df, train_size=train_len)
#%%
# from pmdarima.arima.stationarity import ADFTest

# # Test whether we should difference at the alpha=0.05
# # significance level
# adf_test = ADFTest(alpha=0.05)
# p_val, should_diff = adf_test.should_diff(train)  # (0.01, False)
# #%%
# from pmdarima.arima.utils import ndiffs
# n_adf = ndiffs(train, test='adf')  # -> 0


#%%

model = pm.ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 60))
model.fit(train)

# model = pm.auto_arima(train, p=1, q=2, m=60,
#                              P=0, seasonal=True,
#                              d=0, D=0,Q=0, trace=True,
#                              error_action='ignore',  # don't want to know if an order does not work
#                              suppress_warnings=True,  # don't want convergence warnings
#                              stepwise=True)
print(test.shape[0])
forecasts = model.predict(test.shape[0])
#%%
x = np.arange(df.shape[0])
plt.plot(x[train_len:], df[train_len:], c='blue')
plt.plot(x[train_len:], forecasts, c='green')
plt.show()

# stepwise_model = auto_arima(train, p=1, q=1 , m=48,
                           # P=48, seasonal=True,
                           # d=0, D=0,Q=48, trace=True,
                           # error_action='ignore',  
                           # suppress_warnings=True, 
                           # stepwise=True)
