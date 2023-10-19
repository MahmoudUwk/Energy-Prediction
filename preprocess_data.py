import pandas as pd
import numpy as np
import os
def scaling_input(X,a,b):
    X["P"] = (X["P"] - a) / (b-a)
    return X

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

def sliding_windows2d(data, seq_length, k_step,num_feat):
    x = np.zeros((len(data)-seq_length-k_step+1,seq_length*num_feat))
    y = np.zeros((len(data)-seq_length-k_step+1,k_step))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        x[ind,:] = np.reshape(data[ind:ind+seq_length,:],-1)
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind] = data[ind+seq_length:ind+seq_length+k_step,0]
    return x,y

def slice_data(data, seq_length,k_step):
    #if the data is not divisable by the seq_length+k_step, remove the last few values to make it divisable,...
    #so that all segments are of the same length
    if (len(data)%(seq_length+k_step))!= 0: 
        rem = len(data)%(seq_length+k_step)
        data = data[:-rem]
    data_sliced = np.array(data).reshape(-1,seq_length+k_step)
    return data_sliced[:,:seq_length],np.squeeze(data_sliced[:,seq_length:seq_length+k_step])

    
    
def get_SAMFOR_data(df,seq_length,k_step,percentage_data_use,percentage_train,SARIMA_len,option,SARIMA_pred):
    # df = df["P"]
    df = df[:int(len(df)*percentage_data_use)]
    train_per = percentage_train
    len_data = df.shape[0]
    train_len = int(train_per*len_data)
    train_len_SARIMA = SARIMA_len #int(SARIMA_per*train_len)
    train_len_LSSVR = train_len-train_len_SARIMA
    test_len = len_data - train_len
    a = df["P"][:train_len].min()
    b = df["P"][:train_len].max()
    df_normalized = scaling_input(df,a,b)
    df_normalized = df
    if option == 0:
        return df_normalized[:train_len_SARIMA],train_len_LSSVR,test_len
    
    else:
        SARIMA_linear_pred = np.array(pd.read_csv(SARIMA_pred))
        train_LSSVR = np.array(df_normalized[train_len_SARIMA:train_len_SARIMA+train_len_LSSVR])
        testset = np.array(df_normalized[train_len:])
        del df,df_normalized
        X_LSSVR ,y_LSSVR  = sliding_windows2d(train_LSSVR, seq_length, k_step,train_LSSVR.shape[1])
        X_LSSVR = np.concatenate((X_LSSVR,SARIMA_linear_pred[seq_length:train_len_LSSVR]),axis=1)
        del train_LSSVR
        X_test ,y_test  = sliding_windows2d(testset, seq_length, k_step,testset.shape[1])
        X_test = np.concatenate((X_test,SARIMA_linear_pred[train_len_LSSVR+seq_length-1:train_len_LSSVR+test_len-1]),axis=1)
        del testset
        
        return X_LSSVR,y_LSSVR,X_test,y_test