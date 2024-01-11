# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:43:38 2023

@author: mahmo
"""
import pandas as pd
import os
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None
common_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results_v2'
#%%
results = os.path.join(common_path,'1s/results_LSTM_1s.csv')

df = pd.read_csv(results)

# df.groupby('Algorithm').min()
#RMSE MAE MAPE(%)
df_sorted = df.sort_values(by=['MAE'])
print(df_sorted[:4])

#%%
# results = os.path.join(common_path,'15min/results_15T_seq.csv')
# df = pd.read_csv(results)

# print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))
# #%%
# results = os.path.join(common_path,'30min/results_30T_seq.csv')
# df = pd.read_csv(results)

# print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))