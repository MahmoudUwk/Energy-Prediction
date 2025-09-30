
import matplotlib.pyplot as plt
import os
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, cross_val_score
from keras.layers import Dense,LSTM,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
# from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM
import tensorflow as tf
from preprocess_data2 import *# save_object,RMSE,MAE,MAPE,get_SAMFOR_data,log_results_LSTM,log_results_HOME_C,inverse_transf
from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import numpy as np
from niapy.algorithms.modified import Mod_FireflyAlgorithm
from niapy.algorithms.basic import FireflyAlgorithm

# from niapy.algorithms.basic import BeesAlgorithm

def get_hyperparameters(x):
    """Get hyperparameters for solution `x`."""
    units = int(x[0]*116 + 10)
    num_layers = int(x[1]*6)+1
    seq = int(x[2]*30 + 1)
    lr = x[3]*2e-2 + 0.5e-3
    params =  {
        'units': units,
        'num_layers': num_layers,
        'seq':seq,
        'lr':lr
    }
    # print(params)
    return params

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))

def get_LSTM_model(input_dim,output_dim,units,num_layers, seq,lr,name='LSTM_HP'):
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
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def get_classifier(x,input_dim,output_dim):
    """Get classifier from solution `x`."""
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    return get_LSTM_model(input_dim,output_dim,**params)

def get_data(x,option,datatype_opt):
    if isinstance(x, dict):
        params = x
    else:
        params = get_hyperparameters(x)
    X_train,y_train,X_val,y_val,X_test,y_test,save_path,test_time_axis,scaler = get_SAMFOR_data(option,datatype_opt ,params['seq'])
    return X_train,y_train,X_val,y_val,X_test,y_test,save_path,test_time_axis,scaler

def save_obj(obj,sav_path):
    import pickle
    filehandler = open(sav_path, 'wb') #save the dataset using pickle as an object 
    pickle.dump(obj, filehandler)#saving the dataset
    filehandler.close() #closing the object that saved the dataset

def load_obj(path):
    import pickle
    file_id = open(path,'rb')
    data = pickle.load(file_id)
    file_id.close()
    return data
class LSTMHyperparameterOptimization(Problem):
    def __init__(self, option,datatype_opt,num_epoc):
        super().__init__(dimension=4, lower=0, upper=1)
        self.option = option
        self.datatype_opt = datatype_opt
        self.num_epoc = num_epoc

    def _evaluate(self, x):
        X_train,y_train,X_val,y_val,X_test,y_test,save_path,test_time,scaler = get_data(x,self.option,self.datatype_opt)
        output_dim = 1
        input_dim=(X_train.shape[1],X_train.shape[2])
        y_train = expand_dims(expand_dims(y_train))
        y_test = expand_dims(expand_dims(y_test))
        model = get_classifier(x,input_dim,output_dim)
        out_put_model = [layer.output_shape for c,layer in enumerate(model.layers) if c==len(model.layers)-1][0][1]
        # print(model.summary())
        assert(out_put_model==output_dim)
        # print(X_train.shape,y_train.shape)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
        model.fit(X_train, y_train, epochs=self.num_epoc , batch_size=2**12, verbose=0, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)

        # hp = get_hyperparameters(x)
        # mse = model.evaluate(X_test,y_test)
        # row = ['FF_LSTM_search',mse,mse,mse,hp['seq'],hp['num_layers'],hp['units'],0,self.datatype_opt,0,0]
        # log_results_LSTM(row,self.datatype_opt,save_path)
        return  model.evaluate(X_test,y_test)

option = 3
datatype_opts = ['5T','Home'] #['1s','1T','15T','30T','home','1s']
run_search= 1
pop_size= 5
num_epoc = 2500
FF_itr = 15
alg_range = range(2)
for datatype_opt in datatype_opts:
    for alg_all in alg_range:
        if alg_all == 0:
            # 
            algorithm = Mod_FireflyAlgorithm.Mod_FireflyAlgorithm(population_size = pop_size)
        else:
            algorithm = FireflyAlgorithm(population_size = pop_size)
    #%%
        if run_search: 
            problem = LSTMHyperparameterOptimization(option,datatype_opt,num_epoc)
            task = Task(problem, max_iters=FF_itr, optimization_type=OptimizationType.MINIMIZATION)
    
            
            best_params, best_mse = algorithm.run(task)
            
            best_para_save = get_hyperparameters(best_params)
            


            save_path = get_SAMFOR_data(0,datatype_opt,0,1)
            a_itr,b_itr = task.convergence_data()
            a_eval,b_eval = task.convergence_data(x_axis='evals')
            sav_dict_par = {'a_itr':a_itr,'b_itr':b_itr,'a_eval':a_eval,'b_eval':b_eval,'best_para_save':best_para_save}
            save_obj(sav_dict_par,os.path.join(save_path,'Best_param'+algorithm.Name[0]+'.obj'))
            print('Best parameters:', best_para_save)
            task.plot_convergence(x_axis='evals')
            
            plt.savefig(os.path.join(save_path,'Conv_FF_eval'+str(datatype_opt)+algorithm.Name[0]+'.png'))
            plt.close()
            
            task.plot_convergence()
            
            plt.savefig(os.path.join(save_path,'Conv_FF_itr'+str(datatype_opt)+algorithm.Name[0]+'.png'))
            plt.close()
    
    
        #%%
        save_path = get_SAMFOR_data(0,datatype_opt,0,1)


        best_params = load_obj(os.path.join(save_path,'Best_param'+algorithm.Name[0]+'.obj'))['best_para_save']

        X_train,y_train,X_val,y_val,X_test,y_test,save_path,test_time_axis,scaler = get_data(best_params,option,datatype_opt)

        y_train = expand_dims(expand_dims(y_train))
        input_dim=(X_train.shape[1],X_train.shape[2])
        output_dim = y_train.shape[-1]
        
        model = get_classifier(best_params,input_dim,output_dim)
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)]
        
        history = model.fit(X_train, y_train, epochs=num_epoc, batch_size=2**11, verbose=1, shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks_list)
        

        model.save(algorithm.Name[0]+str(datatype_opt))
        best_epoch = np.argmin(history.history['val_loss'])
     
        y_test = inverse_transf(y_test,scaler)   
        y_test_pred = inverse_transf(model.predict(X_test),scaler)
        rmse = RMSE(y_test,y_test_pred)
        mae = MAE(y_test,y_test_pred)
        mape = MAPE(y_test,y_test_pred)
        print(rmse,mae,mape)
        alg_name = algorithm.Name[0]
        
    #%%
        row = [alg_name,rmse,mae,mape,best_params['seq'],best_params['num_layers'],best_params['units'],best_epoch,datatype_opt,0,0,'all']


        log_results_LSTM(row,datatype_opt,save_path)
        
        # plt.figure(figsize=(20,7),dpi=120)
        # plt.plot(test_time_axis,1000*np.squeeze(y_test), color = 'red', linewidth=2.0, alpha = 0.6)
        # plt.plot(test_time_axis,1000*np.squeeze(y_test_pred), color = 'blue', linewidth=0.8)
        # plt.legend(['Actual','Predicted'])
        # plt.xlabel('Timestamp')
        # plt.xticks( rotation=25 )
        # plt.ylabel('mW')
        # plt.title('Energy Prediction using '+alg_name)
        # plt.show()
        # info_loop = [best_params['seq'],best_params['num_layers'],best_params['units'],best_epoch,datatype_opt]
        # name_sav = ""
        # for n in info_loop:
        #     name_sav = name_sav+str(n)+"_" 
        # plt.savefig(os.path.join(save_path,'LSTM'+name_sav+'.png'))
        # plt.close()
        filename = os.path.join(save_path,alg_name+'.obj')
        obj = {'y_test':y_test,'y_test_pred':y_test_pred}
        save_object(obj, filename)
    
