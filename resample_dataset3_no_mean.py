# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:31:13 2024

@author: mahmo
"""


import pandas as pd


import os
from os import listdir
from os.path import isfile, join
import numpy as np
data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/pwr data paper 2/1Hz"
sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/pwr data paper 2/resampled data"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) and '.csv' in f]


# start_end = []
for counter , file in enumerate(onlyfiles):
    full_path = os.path.join(data_path,file)
    if counter == 0:
        df = pd.read_csv(full_path)

    else:
        df_temp = pd.read_csv(full_path)

        df = pd.concat([df, df_temp])#.sort_values('timestamp').reset_index(drop=True)

rate = 60*30
df = df.sort_values('timestamp').reset_index(drop=True)
df = df.dropna(axis=0)
df_down = df.iloc[::rate,:]
df_down.to_csv(os.path.join(sav_path,'rate_'+str(rate)+'_.csv'),index=False)
#%%
# len_1s = int(0.01*df.shape[0])
# df.iloc[:len_1s,:].to_csv(os.path.join(sav_path,'1s.csv'))
# df.to_csv(os.path.join(sav_path,'1s.csv'))


