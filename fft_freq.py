# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:45:15 2024

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import os
import pandas as pd
from scipy.fft import fft, fftfreq
datatype_opt = 'ele'
path = "C:/Users/Admin/Desktop/New folder/Data/resampled data"
data_path = os.path.join(path,datatype_opt+'.csv')
df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.date), inplace=True,drop=True,append=False)
df.drop(columns=["date"], inplace=True)
# df = df[['P', 'Q', 'V']]
y = np.array(df['hvac_N'].dropna())
T=1
rate = 1
N = len(y)
yf = fft(y)
xf = fftfreq(N, 1 / rate)

plt.plot(xf, np.abs(yf))
plt.show()