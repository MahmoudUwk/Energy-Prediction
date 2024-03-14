from preprocess_data2 import get_SAMFOR_data
# from pmdarima import auto_arima
import pmdarima as pm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

option = 0
datatype_opt = '1s'
seq_length = 7
train_SARIMA_all,train_len_LSSVR,test_len,save_path,time_axis = get_SAMFOR_data(option,datatype_opt,seq_length)
print(train_SARIMA_all.columns)
# feats = ['P', 'Q', 'V', 'I']
#P (order=(2,0,0), seasonal_order=(1, 1, 1, 60),verbose=2)
feats = ['P']
#%%
for feat in feats:
    op = 1
    if op == 1:
        print(train_SARIMA_all[feat].shape)
        print("training start")
        # model = pm.auto_arima(train_SARIMA, m=60,
        #                               seasonal=True,
        #                               trace=True,
        #                               error_action='ignore',  # don't want to know if an order does not work
        #                               suppress_warnings=True,  # don't want convergence warnings
        #                               stepwise=True)
        model = pm.auto_arima(train_SARIMA_all[feat], start_p=0, d=0, start_q=0, max_p=5, max_d=5, max_q=5,
                        start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5,  m=60, #if m=1 seasonal is set to False
                        seasonal=True, error_action='warn', trace=True, suppress_warnings=True,
                        stepwise=True, random_state=20, n_fits=50)
        # model = pm.ARIMA(order=(2,0,0), seasonal_order=(1, 1, 1, 60),verbose=2)
        model.fit(train_SARIMA_all[feat])
        print("training done")
    #%%
        # del train_SARIMA
        forecasts_linear = model.predict(train_len_LSSVR+test_len)
        #%%
        save_name = os.path.join(save_path,'SARIMA_prediction_'+feat+'_.csv')
        np.savetxt(save_name, forecasts_linear, delimiter=",")
        # with open(os.path.join(save_path,'arima.pkl'), 'wb') as pkl:
        #     pickle.dump(model, pkl)
        # print('model_saved')
    
    
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

