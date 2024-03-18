import os
import pandas as pd
from preprocess_data2 import feature_creation
import numpy as np
import seaborn as sns

def feature_creation(data):
    df = data.copy()
    df['Minute'] = data.index.minute
    # df['Second'] = data.index.second
    df['DOW'] = data.index.dayofweek
    df['H'] = data.index.hour
    # df['W'] = data.index.week
    return df
#7332351
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None
path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/pwr data paper 2/resampled data"

datatype_opt = 1
data_type = ['1s_frac','1s','1T','15T','30T']
data_path = os.path.join(path,data_type[datatype_opt]+'.csv')
SARIMA_len = 3600*2
percentage_data_use = 1

df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
#%%
# df = feature_creation(df)
#%%
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
df['P'][:train_len].plot(title="Energy Consumption Data for one sample per second")
sns.heatmap(df.corr(), annot=True, annot_kws={"size": 18})
# sns.heatmap(df.corr())
#%%


df.plot(title="Energy Consumption Data for one sample per second")









