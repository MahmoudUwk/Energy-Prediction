# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:04:26 2023

@author: mahmo
"""
import pandas as pd

data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv"

df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df_downsampled = df.resample('1T').mean()
df_downsampled.to_csv('C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1T.csv')
df_downsampled30 = df.resample('30T').mean()
df_downsampled30.to_csv('C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/30T.csv')
# df.drop(columns=["timestamp"], inplace=True)
# df = df['P']