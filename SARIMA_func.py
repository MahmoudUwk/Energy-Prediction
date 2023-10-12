# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:22:44 2023

@author: msallam
"""
import pmdarima as pm
import pandas as pd
import numpy as np

def train_arima(train,pred_len,p,d,q,P,D,Q,m):
    model = pm.ARIMA(order=(p, d, q), seasonal_order=(P, D, Q, m))
    model.fit(train)
    forecasts = model.predict(pred_len)
    return forecasts

