from preprocess_data import scaling_input
# from pmdarima import auto_arima
import pmdarima as pm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
# data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz/1477227096132.csv"
# save_path = "C:/Users/msallam/Desktop/Kuljeet/results"
data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv"
save_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/results"
df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
#%%
df = df["P"]
df = df[:int(len(df)*0.1)]
train_per = 0.8
len_data = df.shape[0]
train_len = int(train_per*len_data)
train_len_SARIMA = 3600*5#int(SARIMA_per*train_len)
train_len_LSSVR = train_len-train_len_SARIMA
test_len = len_data - train_len
#%%
a = df[:train_len].min()
b = df[:train_len].max()



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
op = 1
if op == 1:
    train_SARIMA = scaling_input(df,a,b)[:train_len_SARIMA]
    print(train_SARIMA.shape)
    del df
    model = pm.ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 60),verbose=2)
    model.fit(train_SARIMA)
    print("training done")
    #%%
    
    with open(os.path.join(save_path,'arima.pkl'), 'wb') as pkl:
        pickle.dump(model, pkl)
    print('model_saved')
#%%
    del train_SARIMA
    forecasts_linear = model.predict(train_len_LSSVR+test_len)
    save_name = os.path.join(save_path,'SARIMA_linear_prediction.csv')
    np.savetxt(save_name, forecasts_linear, delimiter=",")
# model = pm.auto_arima(train, p=1, q=2, m=60,
#                              P=0, seasonal=True,
#                              d=0, D=0,Q=0, trace=True,
#                              error_action='ignore',  # don't want to know if an order does not work
#                              suppress_warnings=True,  # don't want convergence warnings
#                              stepwise=True)

#%%
else:
    with open(os.path.join(save_path,'arima.pkl'), 'rb') as pkl:
        forecasts_linear = pickle.load(pkl).predict(n_periods=train_len_LSSVR+test_len)
    save_name = os.path.join(save_path,'SARIMA_linear_prediction.csv')
    np.savetxt(save_name, forecasts_linear, delimiter=",")

# x = np.arange(df_normalized_SARIMA.shape[0])
# plt.plot(linear_pred, c='blue')
# plt.plot(forecasts_linear, c='green')
# plt.show()
#%%
