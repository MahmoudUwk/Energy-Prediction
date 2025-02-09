
import pandas as pd


import os
from os import listdir
from os.path import isfile, join
import numpy as np

sav_path = "C:/Users/Admin/Desktop/New folder/Data/resampled data"
data_path = "C:/Users/Admin/Desktop/New folder/Data/1Hz"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) and '.csv' in f]

def read_csv(full_path):
    df = pd.read_csv(full_path)
    df.set_index(pd.to_datetime(df.timestamp), inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    return  df

def resample(df,txt,method='mean'):
    if method=='mean':
        df_downsampled = df.resample(txt).mean()
    else:
        df_downsampled = df.resample(txt).max()
    del df
    print(len(np.unique(np.where(df_downsampled.isna())[0])),' Nan values in ',txt)
    df_downsampled = df_downsampled.dropna()
    
    print(df_downsampled.shape)
    return df_downsampled
# start_end = []
# txt_all = ['1T','5T','10T','15T','30T']
txt_all = ['5min']
method = 'mean'
for txt in txt_all:
    for counter , file in enumerate(onlyfiles):
        full_path = os.path.join(data_path,file)
        if counter == 0:
            df = resample(read_csv(full_path),txt,method)
        else:
            df_temp = resample(read_csv(full_path),txt,method)
            df = pd.concat([df, df_temp])
    df.to_csv(os.path.join(sav_path,txt+'_'+method+'.csv'))
#%%
# df.set_index(pd.to_datetime(df.timestamp), inplace=True)
# df.drop(columns=["timestamp"], inplace=True)

# df = df.dropna()
# df.to_csv(os.path.join(sav_path,'1Hz.csv'))
# df = pd.read_csv(data_path)

# resample(df,'1T')

# resample(df,'10T')

# resample(df,'15T')

# resample(df,'30T')

