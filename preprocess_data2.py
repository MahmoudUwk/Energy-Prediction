import pandas as pd
import numpy as np
import os
import pickle
def scaling_input(X,a,b):
    X= (2*((X - a) / (b-a))) - 1
    return X

def RMSE(test,pred):
    return np.sqrt(np.mean((np.squeeze(test) - np.squeeze(pred))**2))

def MAE(test,pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))

def MAPE(test,pred):
    ind = np.where(test!=0)[0].flatten()
    return 100*np.mean(np.abs(np.squeeze(pred[ind]) - np.squeeze(test[ind]))/np.abs(np.squeeze(test[ind])))

def inverse_transf(X,scaler):
    return np.array((X *(scaler.data_max_[0]-scaler.data_min_[0]) )+scaler.data_min_[0])

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

def expand_dims_first(x):
    return np.expand_dims(x,axis=0)

def feature_creation(data):
    df = data.copy()
    df['Minute'] = data.index.minute.astype(float)
    # df['Second'] = data.index.second
    df['DOW'] = data.index.dayofweek.astype(float)
    df['H'] = data.index.hour.astype(float)
    # df['W'] = data.index.week
    return df
#%%
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

def sliding_windows2d_lstm(data, seq_length):
    x = np.zeros((len(data)-seq_length,seq_length,data.shape[1]))
    y = np.zeros((len(data)-seq_length,1))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        x[ind,:,:] = np.squeeze(data[ind:ind+seq_length,:])
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind] = data[ind+seq_length:ind+seq_length+1,0]
    return x,y
#%%
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def plot_test(test_time,y_test,y_test_pred,path,alg_name):
    from matplotlib import pyplot as plt
    scale_mv = 1000
    plt.figure(figsize=(10,7),dpi=180)
    plt.plot(test_time,scale_mv*y_test, color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(test_time,scale_mv*y_test_pred, color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.ylabel('mW')
    plt.title('Energy Prediction using '+alg_name)
    plt.xticks( rotation=25 )
    plt.show()
    plt.savefig(path)



def log_results(row,datatype_opt,save_path):
    save_name = 'results_'+datatype_opt+'.csv'
    cols = ["Algorithm", "RMSE", "MAE", "MAPE(%)","seq","train_time(min)","test_time(s)"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
    
def log_results_LSTM(row,datatype_opt,save_path):

    save_name = 'results_LSTM_feat_3_'+datatype_opt+'.csv'
    cols = ["Algorithm", "RMSE", "MAE", "MAPE(%)","seq","num_layers","units","best epoch","data_type","train_time(min)","test_time(s)","n_feat"]

    df3 = pd.DataFrame(columns=cols)
    if not os.path.isfile(os.path.join(save_path,save_name)):
        df3.to_csv(os.path.join(save_path,save_name),index=False)
        
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
#%%
def load_home_C_data(data2_home_path,sav_path):
    sav_path = os.path.join(sav_path,'Home')
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
    data_path = os.path.join(data2_home_path,'HomeC.csv')

    data = pd.read_csv(data_path).iloc[:-1,:]
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min'))
    data = data.set_index('time')
    data.columns = [i.replace(' [kW]', '') for i in data.columns]
    data['Furnace'] = data[['Furnace 1','Furnace 2']].sum(axis=1)
    data['Kitchen'] = data[['Kitchen 12','Kitchen 14','Kitchen 38']].sum(axis=1) #We could also use the mean 
    data.drop(['Furnace 1','Furnace 2','Kitchen 12','Kitchen 14','Kitchen 38','icon','summary'], axis=1, inplace=True)

    #Replace invalid values in column 'cloudCover' with backfill method
    data['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
    data['cloudCover'] = data['cloudCover'].astype('float')

    #Reorder columns
    # data = data[['use', 'gen', 'House overall', 'Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 'Garage door', 'Barn',
    #              'Well', 'Microwave', 'Living room', 'Furnace', 'Kitchen', 'Solar', 'temperature', 'humidity', 'visibility', 
    #              'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 
    #              'dewPoint', 'precipProbability']]
    data = data[['House overall', 'Furnace', 'Living room', 'Barn', 'temperature', 'humidity',
                   'apparentTemperature', 'pressure', 'cloudCover','windBearing', 'precipIntensity',
                   'dewPoint', 'precipProbability']]
    # data.drop(['use', 'gen'], axis=1, inplace=True)
    # data['month'] = data.index.month
    # data['day'] = data.index.day
    # data['weekday'] = data.index.dayofweek
    # data['hour'] = data.index.hour
    # data['minute'] = data.index.minute
    data = data.resample('30T').mean()
    
    return data,sav_path
    #%%
    
def get_Hzdata(datatype_opt,path,sav_path):

    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
    # data_type = ['1s_frac','1T','15T','30T','home_data','1s','5T']

    
    data_path = os.path.join(path,datatype_opt+'.csv')
    
    df = pd.read_csv(data_path)
    df.set_index(pd.to_datetime(df.timestamp), inplace=True,drop=True,append=False)
    # df.drop(columns=["timestamp"], inplace=True)
    df = df[['P', 'Q', 'V']]
    # df.drop(columns=["I"], inplace=True)
    
    
    sav_path = os.path.join(sav_path,datatype_opt)
    if not os.path.exists(sav_path):
        os.makedirs(sav_path)
    df = feature_creation(df)
    return df,sav_path
    #%%
def get_SAMFOR_data(option,datatype_opt,seq_length,get_sav_path = 0):
    sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/results"
    path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/pwr data paper 2/resampled data"
    if datatype_opt == 'Home':
        df,sav_path = load_home_C_data(path,sav_path)
    else:
        df,sav_path = get_Hzdata(datatype_opt,path,sav_path)
        
    if get_sav_path==1:
        return sav_path
    print(df.columns)
    print('Dataset loaded --- Preprocessing starting')
    if option == 1 or option==0:
        SARIMA_len = 60*12
    else:
        SARIMA_len = 0
    # percentage_data_use = 1
    len_data = df.shape[0]

    k_step = 1
    # df = df[:int(len(df)*percentage_data_use)]

    train_per=0.8
    train_len = int(train_per*len_data)
    
    train_len_SARIMA = SARIMA_len #int(SARIMA_per*train_len)
    train_len_LSSVR = train_len-train_len_SARIMA
    test_len = len_data - train_len

    
    dim = df.ndim
    df_array = np.array(df)
    #%%
    df_normalized = df.copy()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_array[:train_len,:])
    df_normalized.iloc[:,:] = scaler.transform(df_array)
    del df_array,df
    #%%
    if option == 0:
        return df_normalized.iloc[:train_len_SARIMA,:],train_len_LSSVR,test_len,sav_path,scaler
    elif option==2:
        train_clf = np.array(df_normalized[:train_len])
        testset = np.array(df_normalized[train_len:])
        test_time = df_normalized.index[train_len:-seq_length]
        del df_normalized
        if dim>1:
            X_clf ,y_clf  = sliding_windows2d(train_clf, seq_length, k_step,train_clf.shape[1])
        
            X_test ,y_test  = sliding_windows2d(testset, seq_length, k_step,testset.shape[1])
        else:
            X_clf ,y_clf  = sliding_windows(train_clf, seq_length, k_step)
        
            X_test ,y_test  = sliding_windows(testset, seq_length, k_step)

        
        return X_clf,np.squeeze(y_clf),X_test,np.squeeze(y_test),sav_path,test_time,scaler

    elif option==3:
        train_per_and_val_lstm = 0.8
        if datatype_opt == 'Home':
            val_per_lstm = 0.3*train_per_and_val_lstm
        else: 
            val_per_lstm = 0.3*train_per_and_val_lstm
        train_per_lstm = train_per_and_val_lstm - val_per_lstm
        train_len_lstm = int(train_per_lstm*len_data)
        val_len_lstm = int(val_per_lstm*len_data)
        
        #val_len_lstm  train_len_lstm
        train_x = np.array(df_normalized.iloc[:train_len_lstm,:])
        val_x = np.array(df_normalized.iloc[train_len_lstm:train_len_lstm+val_len_lstm,:])
        test_x = np.array(df_normalized.iloc[train_len_lstm+val_len_lstm:,:])
        test_time = df_normalized.index[train_len_lstm+val_len_lstm:-seq_length]
        
 
        X_train ,y_train  = sliding_windows2d_lstm(train_x, seq_length)
        if val_per_lstm != 0:
            X_val ,y_val  = sliding_windows2d_lstm(val_x, seq_length)
        else:
            X_val = []
            y_val = []
        X_test ,y_test  = sliding_windows2d_lstm(test_x, seq_length)
        
        return X_train,np.squeeze(y_train),X_val,np.squeeze(y_val),X_test,np.squeeze(y_test),sav_path,test_time,scaler
    

        
        return X_clf,np.squeeze(y_clf),X_test,np.squeeze(y_test),sav_path,scaler
    
    elif option==1:
        read_path = sav_path
        SARIMA_linear_pred = pd.read_csv(os.path.join(read_path,'SARIMA_prediction_P_.csv'),header=None).to_numpy()
        train_LSSVR = np.array(df_normalized[train_len_SARIMA:train_len_SARIMA+train_len_LSSVR])
        testset = np.array(df_normalized[train_len:])
        test_time = df_normalized.index[train_len:-seq_length]
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
        
        return X_LSSVR,y_LSSVR,X_test,y_test,sav_path,test_time,scaler