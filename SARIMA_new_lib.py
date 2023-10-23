from preprocess_data import get_SAMFOR_data
# from pmdarima import auto_arima
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
save_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models'
option = 0
#%%
op = 1
if op == 1:
    train_SARIMA,train_len_LSSVR,test_len = get_SAMFOR_data(option)
    # train_SARIMA = train_SARIMA[['ds','P']]
    # train_SARIMA = np.squeeze(np.array(train_SARIMA[['P']]))
    print(train_SARIMA.shape)
    # sf = ARIMA(order=(1, 0, 1),season_length=60, seasonal_order=(1, 0, 1))
    sf = AutoARIMA(season_length = 60)
    sf.fit(np.array(train_SARIMA))
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
