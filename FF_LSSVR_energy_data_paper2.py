from sklearn_nature_inspired_algorithms.model_selection import NatureInspiredSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVR
from niapy.algorithms.basic import Mod_FireflyAlgorithm
from niapy.algorithms.basic import *
from lssvr import LSSVR
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import os.path


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

df = pd.read_csv("C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz/1477227096132.csv")
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)

training_size = int(len(df) * 0.7)
#%%
seq_length = 64
k_step = 1
target_col = "P"
X_train , y_train= slice_data(df[target_col][:training_size], seq_length,k_step)
X_test , y_test= slice_data(df[target_col][training_size:], seq_length,k_step)

const_max = X_train.max()
const_min = X_train.min()
X_train = scaling_input(X_train,const_min,const_max)
y_train = scaling_input(y_train,const_min,const_max)
X_test = scaling_input(X_test,const_min,const_max)
y_test = scaling_input(y_test,const_min,const_max)
#%%
# X_train , y_train= sliding_windows(df[target_col][:training_size], seq_length,k_step)
# X_test , y_test= sliding_windows(df[target_col][training_size:], seq_length,k_step)

del df
#%%
algorithm = Mod_FireflyAlgorithm.Mod_FireflyAlgorithm()
# algorithm = FireflyAlgorithm()

algorithms = [Mod_FireflyAlgorithm.Mod_FireflyAlgorithm(),FireflyAlgorithm(), ArtificialBeeColonyAlgorithm(),BatAlgorithm(),
              BeesAlgorithm(),BacterialForagingOptimization(),ClonalSelectionAlgorithm(),
              CuckooSearch(),CatSwarmOptimization(),ForestOptimizationAlgorithm(),
              FlowerPollinationAlgorithm(),BareBonesFireworksAlgorithm(),
              GravitationalSearchAlgorithm(),GlowwormSwarmOptimization(),GreyWolfOptimizer(),HarrisHawksOptimization(),
              HarmonySearch(),KrillHerd(),MonarchButterflyOptimization(),MothFlameOptimizer(),ParticleSwarmAlgorithm()
             , SineCosineAlgorithm()]

#%%
param_grid = { 
    'C': [10**c for c in range(-3,3)], 
    'gamma': [10**g for g in range(-3,3)],
}
#%%
cols = ["Algorithm", "RMSE", "MAE", "MAPE", "time (s)","C","gamma"]
df3 = pd.DataFrame(columns=cols)
if not os.path.isfile('search_alg_results_50.csv'):
    df3.to_csv('search_alg_results_50.csv',index=False)
    #%%
for alg in range(len(algorithms)):
 
    clf = LSSVR(kernel='rbf')
    
    nia_search = NatureInspiredSearchCV(
        clf,
        param_grid,
        algorithm = algorithms[alg], # hybrid bat algorithm
        population_size=50,
        max_n_gen=25,
        max_stagnating_gen=10,
        runs=3,
        random_state=None, # or any number if you want same results on each run
    )
    start_time = time.time()
    nia_search.fit(X_train, y_train)
    end_time = time.time()
    time_passed = end_time - start_time
    # the best params are stored in nia_search.best_params_
    # finally you can train your model with best params from nia search
    #%%

    print("best parameters",nia_search.best_params_)
    new_clf = LSSVR(**nia_search.best_params_)
    new_clf.fit(X_train, y_train)
    
    y_test_pred = new_clf.predict(X_test).reshape(-1,1)
    #%%
    plt.figure(figsize=(10,5))
    plt.plot(y_test, color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.show()
    plt.savefig(algorithms[alg].Name[0])
    plt.close()
    rmse = RMSE(y_test,y_test_pred)
    mae = MAE(y_test,y_test_pred)
    mape = MAPE(y_test,y_test_pred)
    print('LLSVR FF:',rmse)
    row = [algorithms[alg].Name[0],rmse,mae,mape,time_passed,nia_search.best_params_['C'],nia_search.best_params_['gamma']]
    #%%
    df = pd.read_csv('search_alg_results_50.csv')
    df.loc[len(df)] = row
    print(df)
    df.to_csv('search_alg_results_50.csv',mode='w', index=False,header=True)
#%%

# df2 = pd.read_csv('search_alg_results.csv')
# # plt.figure(figsize=(10,5))
# data = df2['time (s)']
# x = range(len(data))
# fig, ax = plt.scatter(x,data, color = 'blue', linewidth=2.0, alpha = 0.6)

# # plt.plot(y_test_pred, color = 'blue', linewidth=0.8)
# plt.xlabel('Algorithm')
# plt.xlabel('Time')
# for i, txt in enumerate(list(df2['Algorithm'])):
#     ax.annotate(txt, (x[i],data[i]))
# plt.show()
# plt.savefig(algorithms[alg].Name[0])
# plt.close()











