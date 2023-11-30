from preprocess_data import get_SAMFOR_data
# from pmdarima import auto_arima
import pmdarima as pm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

option = 0
datatype_opt = 0
seq_length = 6
train_SARIMA,train_len_LSSVR,test_len,save_path = get_SAMFOR_data(option,datatype_opt,seq_length)
op = 1
if op == 1:
    print(train_SARIMA.shape)
    print("training start")
    model = pm.auto_arima(train_SARIMA, m=60,
                                  seasonal=True,
                                  trace=True,
                                  error_action='ignore',  # don't want to know if an order does not work
                                  suppress_warnings=True,  # don't want convergence warnings
                                  stepwise=True)
    # model = pm.ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 30),verbose=2)
    model.fit(train_SARIMA)
    print("training done")
#%%
    # del train_SARIMA
    forecasts_linear = model.predict(train_len_LSSVR+test_len)
    #%%
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
