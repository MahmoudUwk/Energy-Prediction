from preprocess_data import get_SAMFOR_data
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
#%%dfsdf
seq_length = 6
percentage_data_use = 0.15
k_step = 1
percentage_train = 0.8
SARIMA_len = 3600
option = 0
#%%
train_SARIMA,train_len_LSSVR,test_len = get_SAMFOR_data(df,seq_length,k_step,percentage_data_use,percentage_train,SARIMA_len,option,SARIMA_pred='')
del df
op = 1
if op == 1:
    print(train_SARIMA.shape)
    # model = pm.auto_arima(train_SARIMA['P'], p=1,p_max=3, q=2, m=60,
    #                               P=0, seasonal=True,
    #                               d=0, D=0,Q=0, trace=True,
    #                               error_action='ignore',  # don't want to know if an order does not work
    #                               suppress_warnings=True,  # don't want convergence warnings
    #                               stepwise=True)
    model = pm.ARIMA(order=(1, 0, 1), seasonal_order=(2, 0, 2, 60),verbose=2)
    model.fit(train_SARIMA['P'])
    print("training done")
#%%
    del train_SARIMA
    forecasts_linear = model.predict(train_len_LSSVR+test_len)
    save_name = os.path.join(save_path,'SARIMA_linear_prediction.csv')
    np.savetxt(save_name, forecasts_linear, delimiter=",")
    with open(os.path.join(save_path,'arima.pkl'), 'wb') as pkl:
        pickle.dump(model, pkl)
    print('model_saved')


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
