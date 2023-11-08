# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:04:26 2023

@author: mahmo
"""
import pandas as pd


import os
from os import listdir
from os.path import isfile, join

data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz"
# data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz"

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) and '.csv' in f]


for counter , file in enumerate(onlyfiles):
    full_path = os.path.join(data_path,file)
    if counter == 0:
        df = pd.read_csv(full_path)
    else:
        df.append(pd.read_csv(full_path))
#%%
# df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df_downsampled = df.resample('1T').mean()
df_downsampled.to_csv(os.path.join(data_path,'1T.csv'))

df_downsampled10 = df.resample('10T').mean()
df_downsampled10.to_csv(os.path.join(data_path,'10T.csv'))

df_downsampled15 = df.resample('15T').mean()
df_downsampled15.to_csv(os.path.join(data_path,'15T.csv'))


df_downsampled30 = df.resample('30T').mean()
df_downsampled30.to_csv(os.path.join(data_path,'30T.csv'))
# df.drop(columns=["timestamp"], inplace=True)
# df = df['P']