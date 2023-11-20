# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:43:38 2023

@author: mahmo
"""
import pandas as pd
import os
common_path = 'C:/Users/msallam/Desktop/Energy Prediction/results'
#%%
results = os.path.join(common_path,'1min/results_1T_seq.csv')

df = pd.read_csv(results)

df.groupby('Algorithm').min()
print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))

#%%
results = os.path.join(common_path,'15min/results_15T_seq.csv')
df = pd.read_csv(results)

print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))
#%%
results = os.path.join(common_path,'30min/results_30T_seq.csv')
df = pd.read_csv(results)

print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))