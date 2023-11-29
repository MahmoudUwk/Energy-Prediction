# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:58:43 2023

@author: mahmo
"""
import os
import pandas as pd
from preprocess_data import feature_creation
import numpy as np
import seaborn as sns
path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/resampled data"
datatype_opt = 0
data_type = ['1T','15T','30T']
data_path = os.path.join(path,data_type[datatype_opt]+'.csv')
SARIMA_len = 3600*2
percentage_data_use = 1

df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)

 
k_step = 1
df = df[:int(len(df)*percentage_data_use)]
train_per = 0.8
len_data = df.shape[0]
train_len = int(train_per*len_data)
train_len_SARIMA = SARIMA_len #int(SARIMA_per*train_len)
train_len_LSSVR = train_len-train_len_SARIMA
test_len = len_data - train_len
# df = feature_creation(df)
dim = df.ndim
df_array = np.array(df)

df_normalized = df.copy()
#%%
# df['P'][:train_len].plot(title="Energy Consumption Data for one sample per minute")
sns.heatmap(df.corr(), annot=True, annot_kws={"size": 18})
# sns.heatmap(df.corr())