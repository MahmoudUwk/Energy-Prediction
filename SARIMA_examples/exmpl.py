# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:24:47 2023

@author: mahmo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
# print(os.listdir("../input"))

# Read our data
data = pd.read_csv('IPG2211A2N.csv',index_col=0)
data.head()

# Change our data index from string to datetime
data.index = pd.to_datetime(data.index)
data.columns = ['Energy Production']
data.head()

# Import Plotly & Cufflinks libraries and run it in Offline mode
import plotly.offline as py
py.init_notebook_mode(connected=True)
py.enable_mpl_offline()

import cufflinks as cf
cf.go_offline()

# Now, plot our time serie
data.iplot(title="Energy Production Between Jan 1939 to May 2019")

# We'll use statsmodels to perform a decomposition of this time series
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model='multiplicative')

fig = result.plot()

py.iplot_mpl(fig)
# Try "py.plot_mpl(fig)" on your local Anaconda, it'll show greater plot than this one
#SARIMA(p, d, q)×(P, D, Q)S, #  p is the nonseasonal AR,  d is nonseasonal diferencing,  q is the nonseasonal MA
# P is the seasonal AR, D is seasonal diferencing,  Q is the seasonal MA order,  S is the season length
#SARIMA (1, 0, 1)×(48, 0, 48)48
# The Pmdarima library for Python allows us to quickly perform this grid search 
from pmdarima import auto_arima

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())