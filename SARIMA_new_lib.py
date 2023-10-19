from preprocess_data import get_SAMFOR_data
# from pmdarima import auto_arima
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA 
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
df = df.rename(columns={'timestamp': 'ds'})
df.set_index(pd.to_datetime(df.ds), inplace=True)
# df.index.name='unique_id'
# df.drop(columns=["ds"], inplace=True)
#%%
seq_length = 6
percentage_data_use = 0.4
k_step = 1
percentage_train = 0.8
SARIMA_len = 3600*5
option = 0
#%%
op = 1
if op == 1:
    train_SARIMA,train_len_LSSVR,test_len = get_SAMFOR_data(df,seq_length,k_step,percentage_data_use,percentage_train,SARIMA_len,option,SARIMA_pred_path='')
    # train_SARIMA = train_SARIMA[['ds','P']]
    train_SARIMA = np.squeeze(np.array(train_SARIMA[['P']]))
    print(train_SARIMA.shape)
    del df
    # sf = ARIMA(order=(1, 0, 1),season_length=60, seasonal_order=(1, 0, 1))
    sf = AutoARIMA(season_length = 60)
    sf.fit(train_SARIMA)
    # model = pm.ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 60),verbose=2)
    # model.fit(train_SARIMA)
    print("training done")
    
    with open(os.path.join(save_path,'arima_2.pkl'), 'wb') as pkl:
        pickle.dump(sf, pkl)
    print('model_saved')
#%%
    del train_SARIMA
    forecasts_linear = sf.predict(h=train_len_LSSVR+test_len)
    save_name = os.path.join(save_path,'SARIMA_linear_prediction_2.csv')
    np.savetxt(save_name, forecasts_linear['mean'], delimiter=",")
# model = pm.auto_arima(train, p=1,p_max=3, q=2, m=60,
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
