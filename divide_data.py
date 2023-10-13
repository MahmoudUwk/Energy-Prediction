from preprocess_data import scaling_input,sliding_windows
# from pmdarima import auto_arima
import pmdarima as pm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz/1477227096132.csv"
# save_path = "C:/Users/msallam/Desktop/Kuljeet/results"
data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv"
save_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/results"
df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
#%%
df = df["P"]
train_per = 0.8
len_data = df.shape[0]
train_len = int(train_per*len_data)
train_len_SARIMA = 3600*5#int(SARIMA_per*train_len)
train_len_LSSVR = train_len-train_len_SARIMA
test_len = len_data - train_len
#%%
a = df[:train_len].min()
b = df[:train_len].max()
df_normalized = scaling_input(df,a,b)
train_SARIMA = df_normalized[:train_len_SARIMA]
train_LSSVR = df_normalized[train_len_SARIMA:train_len_SARIMA+train_len_LSSVR]
testset = df_normalized[train_len:]
del df,df_normalized
#%%
seq_length = 6
k_step = 1
X_LSSVR ,y_LSSVR  = sliding_windows(train_LSSVR, seq_length, k_step)
del train_LSSVR
X_test ,y_test  = sliding_windows(testset, seq_length, k_step)
del testset