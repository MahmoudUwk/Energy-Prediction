from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from keras.callbacks import EarlyStopping
from keras.layers import  LSTM, BatchNormalization,Dense#,Bidirectional
from keras.models import  Sequential #,load_model
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# import tensorflow as tf
import os
# import keras
from preprocess_data2 import *#RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM,log_results_HOME_C,inverse_transf

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

def get_LSTM_model(units,input_dim,output_dim,num_layers,name='LSTM_HP'):
    model = Sequential(name=name)
    flag_seq = True
    if num_layers == 1:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = False))
    else:
        model.add(LSTM(units=units,  input_shape=input_dim,return_sequences = True))
    for dummy in range(num_layers-1):
        if dummy == num_layers-2:
            flag_seq = False     
        model.add(LSTM(units=units,return_sequences = flag_seq))
    model.add(Dense(output_dim))
    return model


#%%
option = 3
alg_name = 'LSTM'
data_types = ['1s']
seq_all = [7]#[5,20]
num_units = [15]#[35]#[8,10,15]
num_layers_all = [1]
epochs_num = 2000
# lr = 0.010164565169640837
lr=0.001
adam=Adam(learning_rate=lr)
# rmspr = RMSprop()
opt_chosen = adam
num_feat = [6]
val_split_size=0
for datatype_opt in data_types:
    for n_feat in num_feat:
        for seq_counter , seq in enumerate(seq_all):
            X_train,y_train,X_val,y_val,X_test,y_test,save_path,test_time_axis,scaler = get_SAMFOR_data(option,datatype_opt,seq)

            # X_train = X_train[:,:,:n_feat]
            # X_val = X_val[:,:,:n_feat]
            # X_test = X_test[:,:,:n_feat]

        
            y_test = inverse_transf(y_test,scaler)
    
            print(X_train.shape)
            y_train = expand_dims(expand_dims(y_train))
            y_val = expand_dims(expand_dims(y_val))
            if len(X_train.shape)<3:
                X_train = expand_dims(X_train)
                X_val = expand_dims(X_val)
                X_test = expand_dims(X_test)
            print(X_train.shape,X_val.shape,X_test.shape)
            #%% LSTM model
    
            
            drop_out = 0
            callback_falg = 1
            input_dim=(X_train.shape[1],X_train.shape[2])
            output_dim = y_train.shape[-1]
            batch_size_n = 2**10
            for units in num_units:
                print(input_dim,output_dim)
                for num_layers in num_layers_all:    
                    model = get_LSTM_model(units,input_dim,output_dim,num_layers)            
                    model.compile(optimizer=opt_chosen, loss='mse')
                    # model.summary()
                    # ,callbacks=callbacks_list
                    print('start training')
                    start_train = time.time()
                    if callback_falg:
                        callbacks_list = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
                        if len(X_val) ==0:
                            history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=0, shuffle=True, validation_split=val_split_size,callbacks=callbacks_list)
                        else:
                            history = model.fit(X_train, y_train, epochs=epochs_num, batch_size=batch_size_n, verbose=0, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
                    else:
                        print('Stop')
                    #%%
                    end_train = time.time()
                    print('End training')
                    train_time = (end_train - start_train)/60
                    if callback_falg and len(X_val) !=0:
                        best_epoch =np.argmin(history.history['val_loss'])
                    else:
                        best_epoch = 0
                    # model.save(filepath)
                    #%%
                    
                    start_test = time.time()
                    y_test_pred = inverse_transf(model.predict(X_test),scaler)
                    end_test = time.time()
                    test_time = end_test - start_test
                    #%%
                    rmse = RMSE(y_test,y_test_pred)
                    mae = MAE(y_test,y_test_pred)
                    mape = MAPE(y_test,y_test_pred)
                    print(rmse,mae,mape)
                    # best_epoch = epochs_num
                   
                    row = [alg_name,rmse,mae,mape,seq,num_layers,units,best_epoch,datatype_opt,train_time,test_time,n_feat]
        
                    log_results_LSTM(row,datatype_opt,save_path)
                    #%%
                    # plt.figure(figsize=(10,7),dpi=180)
                    # plt.plot(test_time_axis,1000*np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
                    # plt.plot(test_time_axis,1000*np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
                    # plt.legend(['Actual','Predicted'])
                    # plt.xlabel('Timestamp')
                    # plt.xticks( rotation=25)
                    # plt.ylabel('mW')
                    # plt.title('Energy Prediction using '+alg_name)
                    # plt.show()
                    # info_loop = [seq,num_layers,units,best_epoch,datatype_opt,n_feat]
                    # name_sav = ""
                    # for n in info_loop:
                    #     name_sav = name_sav+str(n)+"_" 
                    # plt.savefig(os.path.join(save_path,'LSTM'+name_sav+'.png'))
                    plt.close()
                    filename = os.path.join(save_path,alg_name+'.obj')
                    obj = {'y_test':y_test,'y_test_pred':y_test_pred}
                    save_object(obj, filename)
#%%
# save_name = 'results_LSTM_feat_3_1s.csv'

# df = pd.read_csv(os.path.join(save_path,save_name))
# print(df['RMSE'].min())
# print(df.sort_values(by=['RMSE'])[:5])
# RMSE_all = df.groupby(by='n_feat').min()['RMSE'].mean() 
# print(RMSE_all)
# l_u = []
# for n,cluster_dat in df.groupby(by='n_feat'):
#     (a,b) = cluster_dat[['num_transformer_blocks','mlp_units']].iloc[cluster_dat['RMSE'].argmin()]
#     l_u.append((a,b))