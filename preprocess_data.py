import pandas as pd
import numpy as np
import os
def scaling_input(X,a,b):
    X= (2*((X - a) / (b-a))) - 1
    return X

def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(test) - np.squeeze(pred))**2))

def MAE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))

def MAPE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test))/np.abs(np.squeeze(test)))

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

def log_results(row):
    save_path = 'C:/Users/msallam/Desktop/Kuljeet/results'
    # save_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models'
    save_name = 'results_50Hz_10seq.csv'
    cols = ["Algorithm", "RMSE", "MAE", "MAPE"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    
def get_SAMFOR_data(option):
    # path = "C:/Users/msallam/Desktop/Kuljeet/"
    path = "C:/Users/msallam/Desktop/Kuljeet/resampled data"
    data_path = os.path.join(path,'30T.csv')
    SARIMA_len = 1500
    percentage_data_use = 1
    df = pd.read_csv(data_path)
    df.set_index(pd.to_datetime(df.timestamp), inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    df = df['P']
    
    seq_length = 6
    
    k_step = 1
    percentage_train = 0.8
    
    

    df = df[:int(len(df)*percentage_data_use)]
    train_per = percentage_train
    len_data = df.shape[0]
    train_len = int(train_per*len_data)
    train_len_SARIMA = SARIMA_len #int(SARIMA_per*train_len)
    train_len_LSSVR = train_len-train_len_SARIMA
    test_len = len_data - train_len
    dim = df.ndim
    if dim>1:
        a = df.iloc[:train_len,0].min()
        b = df.iloc[:train_len,0].max()
    else:
        a = df.iloc[:train_len].min()
        b = df.iloc[:train_len].max()
    df_normalized = scaling_input(df,a,b)
    if option == 0:
        return df_normalized[:train_len_SARIMA],train_len_LSSVR,test_len
    elif option==2:
        train_clf = np.array(df_normalized[:train_len_SARIMA+train_len_LSSVR])
        testset = np.array(df_normalized[train_len:])
        del df,df_normalized
        if dim>1:
            X_clf ,y_clf  = sliding_windows2d(train_clf, seq_length, k_step,train_clf.shape[1])
        
            X_test ,y_test  = sliding_windows2d(testset, seq_length, k_step,testset.shape[1])
        else:
            X_clf ,y_clf  = sliding_windows(train_clf, seq_length, k_step)
        
            X_test ,y_test  = sliding_windows(testset, seq_length, k_step)

        
        return X_clf,y_clf,X_test,y_test        
    
    elif option==1:
        # SARIMA_pred = os.path.join("C:/Users/mahmo/OneDrive/Desktop/kuljeet/results",'SARIMA_linear_prediction.csv')
        
        SARIMA_pred = 'C:/Users/msallam/Desktop/Kuljeet/results/SARIMA_linear_prediction.csv'
        SARIMA_linear_pred = np.array(pd.read_csv(SARIMA_pred))
        train_LSSVR = np.array(df_normalized[train_len_SARIMA:train_len_SARIMA+train_len_LSSVR])
        testset = np.array(df_normalized[train_len:])
        del df,df_normalized
        if dim>1:
            X_LSSVR ,y_LSSVR  = sliding_windows2d(train_LSSVR, seq_length, k_step,train_LSSVR.shape[1])
            X_test ,y_test  = sliding_windows2d(testset, seq_length, k_step,testset.shape[1])
        else:
            X_LSSVR ,y_LSSVR  = sliding_windows(train_LSSVR, seq_length, k_step)
            X_test ,y_test  = sliding_windows(testset, seq_length, k_step)
        X_LSSVR = np.concatenate((X_LSSVR,SARIMA_linear_pred[seq_length:train_len_LSSVR]),axis=1)
        del train_LSSVR
        X_test = np.concatenate((X_test,SARIMA_linear_pred[train_len_LSSVR+seq_length-1:train_len_LSSVR+test_len-1]),axis=1)
        del testset
        
        return X_LSSVR,y_LSSVR,X_test,y_test