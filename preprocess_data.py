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

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

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

def sliding_windows2d_lstm(data, seq_length, k_step,num_feat):
    x = np.zeros((len(data)-seq_length-k_step+1,seq_length,num_feat))
    y = np.zeros((len(data)-seq_length-k_step+1,k_step))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        x[ind,:,:] = np.squeeze(data[ind:ind+seq_length,:])
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind] = data[ind+seq_length:ind+seq_length+k_step,0]
    return x,y

def sliding_windows2d_lstm_arima(data, seq_length, k_step,num_feat,P_arima,Q_arima,V_arima,I_arima):
    x = np.zeros((len(data)-seq_length-k_step+1,seq_length+1,num_feat))
    y = np.zeros((len(data)-seq_length-k_step+1,k_step))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        one_step = np.arange(ind+seq_length,ind+seq_length+k_step)
        arima_step = np.append(np.array([P_arima[one_step],Q_arima[one_step],V_arima[one_step],I_arima[one_step]]),np.squeeze(data[one_step,4:]))
        arima_step = np.expand_dims(arima_step, axis = 0)
        x[ind,:,:] = np.concatenate((np.squeeze(data[ind:ind+seq_length,:]),arima_step),axis=0)
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind] = data[one_step,0]
    return x,y

def slice_data(data, seq_length,k_step):
    #if the data is not divisable by the seq_length+k_step, remove the last few values to make it divisable,...
    #so that all segments are of the same length
    if (len(data)%(seq_length+k_step))!= 0: 
        rem = len(data)%(seq_length+k_step)
        data = data[:-rem]
    data_sliced = np.array(data).reshape(-1,seq_length+k_step)
    return data_sliced[:,:seq_length],np.squeeze(data_sliced[:,seq_length:seq_length+k_step])

def log_results(row,datatype_opt,save_path):
    data_type = ['1s','1T','15T','30T']
    save_name = 'results_'+data_type[datatype_opt]+'.csv'
    cols = ["Algorithm", "RMSE", "MAE", "MAPE","seq"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    
def log_results_LSTM(row,datatype_opt,save_path):
    data_type = ['1s','1T','15T','30T']
    save_name = 'results_LSTM_'+data_type[datatype_opt]+'.csv'
    cols = ["Algorithm", "RMSE", "MAE", "MAPE","seq","num_layers","units","best epoch","data_type"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    #%%
def feature_creation(data):
    df = data.copy()
    df['Minute'] = data.index.minute
    # df['Second'] = data.index.second
    df['DOW'] = data.index.dayofweek
    df['H'] = data.index.hour
    # df['W'] = data.index.week
    return df
    #%%
def get_SAMFOR_data(option,datatype_opt,seq_length):
    path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/resampled data"
    sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/results_v2"
    data_type = ['1s_frac','1T','15T','30T']
    data_sav= ['1s','1T','15T','30T']
    data_path = os.path.join(path,data_type[datatype_opt]+'.csv')
    sav_path = os.path.join(sav_path,data_sav[datatype_opt])
    SARIMA_len_all = [60*12,60*12,4*60,2*120]
    SARIMA_len = SARIMA_len_all[datatype_opt]
    pu_all = [1,1,1,1]
    percentage_data_use = pu_all[datatype_opt]
    
    df = pd.read_csv(data_path)
    df.set_index(pd.to_datetime(df.timestamp), inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
 
    k_step = 1
    df = df[:int(len(df)*percentage_data_use)]
    train_per = 0.8
    len_data = df.shape[0]
    train_len = int(train_per*len_data)
    train_len_SARIMA = SARIMA_len #int(SARIMA_per*train_len)
    train_len_LSSVR = train_len-train_len_SARIMA
    test_len = len_data - train_len
    df = feature_creation(df)
    dim = df.ndim
    df_array = np.array(df)
    #%%
    df_normalized = df.copy()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df_array)
    df_normalized.iloc[:,:] = scaler.transform(df_array)
    del df_array,df
    if option == 0:
        return df_normalized.iloc[:train_len_SARIMA,:],train_len_LSSVR,test_len,sav_path
    elif option==2:
        train_clf = np.array(df_normalized[:train_len])
        testset = np.array(df_normalized[train_len:])
        del df_normalized
        if dim>1:
            X_clf ,y_clf  = sliding_windows2d(train_clf, seq_length, k_step,train_clf.shape[1])
        
            X_test ,y_test  = sliding_windows2d(testset, seq_length, k_step,testset.shape[1])
        else:
            X_clf ,y_clf  = sliding_windows(train_clf, seq_length, k_step)
        
            X_test ,y_test  = sliding_windows(testset, seq_length, k_step)

        
        return X_clf,np.squeeze(y_clf),X_test,np.squeeze(y_test),sav_path

    elif option==3:
        train_clf = np.array(df_normalized[:train_len])
        testset = np.array(df_normalized[train_len:])
        dim = len(train_clf.shape)
        del df_normalized
        if dim>1:
            X_clf ,y_clf  = sliding_windows2d_lstm(train_clf, seq_length, k_step,train_clf.shape[1])
        
            X_test ,y_test  = sliding_windows2d_lstm(testset, seq_length, k_step,testset.shape[1])
        else:
            X_clf ,y_clf  = sliding_windows(train_clf, seq_length, k_step)
        
            X_test ,y_test  = sliding_windows(testset, seq_length, k_step)

        
        return X_clf,np.squeeze(y_clf),X_test,np.squeeze(y_test),sav_path
    
    elif option==4:
        ind_train = np.arange(train_len_SARIMA,train_len_SARIMA+train_len_LSSVR)
        train_clf = np.array(df_normalized.iloc[ind_train,:])
        sarima_ind = np.arange(0,len(ind_train))
        # sarima_ind_test = np.arange(train_len_LSSVR+seq_length-1,train_len_LSSVR+test_len-1)
        testset = np.array(df_normalized[train_len:])
        dim = len(train_clf.shape)
        feats = ['P', 'Q', 'V', 'I']
        read_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results_v2\\1s'
        P_arima = pd.read_csv(os.path.join(read_path,'SARIMA_prediction_'+feats[0]+'_.csv'),header=None).to_numpy().squeeze()
        Q_arima = pd.read_csv(os.path.join(read_path,'SARIMA_prediction_'+feats[1]+'_.csv'),header=None).to_numpy().squeeze()
        V_arima = pd.read_csv(os.path.join(read_path,'SARIMA_prediction_'+feats[2]+'_.csv'),header=None).to_numpy().squeeze()
        I_arima = pd.read_csv(os.path.join(read_path,'SARIMA_prediction_'+feats[3]+'_.csv'),header=None).to_numpy().squeeze()
        
        # del df_normalized

        X_clf ,y_clf  = sliding_windows2d_lstm_arima(train_clf, seq_length, k_step,train_clf.shape[1],P_arima[sarima_ind],Q_arima[sarima_ind],V_arima[sarima_ind],I_arima[sarima_ind])
    
        X_test ,y_test  = sliding_windows2d_lstm_arima(testset, seq_length, k_step,testset.shape[1],P_arima[len(ind_train):],Q_arima[len(ind_train):],V_arima[len(ind_train):],I_arima[len(ind_train):])


        
        return X_clf,np.squeeze(y_clf),X_test,np.squeeze(y_test),sav_path
    
    elif option==1:
        read_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results_v2\\1s'
        SARIMA_linear_pred = pd.read_csv(os.path.join(read_path,'SARIMA_prediction_P_.csv'),header=None).to_numpy().squeeze()
        train_LSSVR = np.array(df_normalized[train_len_SARIMA:train_len_SARIMA+train_len_LSSVR])
        testset = np.array(df_normalized[train_len:])
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
        
        return X_LSSVR,y_LSSVR,X_test,y_test,sav_path