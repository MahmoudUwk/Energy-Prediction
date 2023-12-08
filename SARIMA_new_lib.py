from preprocess_data import get_SAMFOR_data
# from pmdarima import auto_arima
# from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA 
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt
# import pickle


option = 0
datatype_opt = 0
seq_length = 6
train_SARIMA_all,train_len_LSSVR,test_len,save_path = get_SAMFOR_data(option,datatype_opt,seq_length)
print(train_SARIMA_all.columns)
feats = ['P', 'Q', 'V', 'I']
for feat in feats:
    train_SARIMA = train_SARIMA_all[feat]
    print(train_SARIMA.shape)
    # sf = ARIMA(order=(1, 0, 1),season_length=60, seasonal_order=(1, 0, 1))
    sf = AutoARIMA(season_length = 60)
    sf.fit(train_SARIMA)
    

    forecasts_linear = sf.predict(h=train_len_LSSVR+test_len)
    save_name = os.path.join(save_path,'SARIMA_prediction_'+feat+'_.csv')
    np.savetxt(save_name, forecasts_linear, delimiter=",")


