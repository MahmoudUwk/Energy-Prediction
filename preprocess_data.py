import pandas as pd
import numpy as np

def scaling_input(X,a,b):
    return (X - a) / (b-a)

def RMSE(test,pred):
    return np.sqrt(np.mean((test - pred)**2))

def MAE(test,pred):
    return np.mean(np.abs(pred - test))

def MAPE(test,pred):
    return np.mean(np.abs(pred - test)/np.abs(test))

def sliding_windows(data, seq_length, k_step):
    x = np.zeros((len(data)-seq_length-k_step+1,seq_length))
    y = np.zeros((len(data)-seq_length-k_step+1,k_step))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        x[ind,:] = data[ind:ind+seq_length]
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind,:] = data[ind+seq_length:ind+seq_length+k_step]
    return x,y

def slice_data(data, seq_length,k_step):
    #if the data is not divisable by the seq_length+k_step, remove the last few values to make it divisable,...
    #so that all segments are of the same length
    if (len(data)%(seq_length+k_step))!= 0: 
        rem = len(data)%(seq_length+k_step)
        data = data[:-rem]
    data_sliced = np.array(data).reshape(-1,seq_length+k_step)
    return data_sliced[:,:seq_length],np.squeeze(data_sliced[:,seq_length:seq_length+k_step])

def preprocess(data_path,percentage_train,seq_length,k_step,slide_or_slice):
    df = pd.read_csv(data_path)
    df.set_index(pd.to_datetime(df.timestamp), inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    
    training_size = int(len(df) *percentage_train )
    #%%
    target_col = "P"
    if slide_or_slice == 0:
        X_train , y_train= sliding_windows(df[target_col][:training_size], seq_length,k_step)
        X_test , y_test= sliding_windows(df[target_col][training_size:], seq_length,k_step)       
    else:       
        X_train , y_train= slice_data(df[target_col][:training_size], seq_length,k_step)
        X_test , y_test= slice_data(df[target_col][training_size:], seq_length,k_step)
    
    const_max = X_train.max()
    const_min = X_train.min()
    X_train = scaling_input(X_train,const_min,const_max)
    y_train = scaling_input(y_train,const_min,const_max)
    X_test = scaling_input(X_test,const_min,const_max)
    y_test = scaling_input(y_test,const_min,const_max)
        
    return X_train,y_train,X_test,y_test,const_max,const_min