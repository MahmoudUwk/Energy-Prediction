# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:43:38 2023

@author: mahmo
"""
import pandas as pd
results = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models/1T/results_1T.csv'

df = pd.read_csv(results)

df.groupby('Algorithm').min()
print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))

#%%
results = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models/Seq 6/results.csv'

df = pd.read_csv(results)

print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))
#%%
results = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models/Seq 8/results.csv'

df = pd.read_csv(results)

print(df.groupby('Algorithm').min().sort_values(by=['RMSE']))