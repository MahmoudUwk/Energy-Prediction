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
import os
from preprocess_data import preprocess,RMSE,MAE,MAPE


data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz/1477227096132.csv"
save_path = "C:/Users/msallam/Desktop/Kuljeet/results"
percentage_train = 0.7
seq_length = 10
k_step = 1
slide_or_slice = 1 #0 for slide and  1 for slice
X_train,y_train,X_test,y_test,const_max,const_min = preprocess(data_path,percentage_train,seq_length,k_step,slide_or_slice)

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
if not os.path.isfile(os.path.join(save_path,'search_alg_results.csv')):
    df3.to_csv(os.path.join(save_path,'search_alg_results.csv'),index=False)
    #%%
n_pop = 25
itr = 10
for alg in range(len(algorithms)):
 
    clf = LSSVR(kernel='rbf')
    
    nia_search = NatureInspiredSearchCV(
        clf,
        param_grid,
        algorithm = algorithms[alg], # hybrid bat algorithm
        population_size=n_pop,
        max_n_gen=itr,
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
    df = pd.read_csv(os.path.join(save_path,'search_alg_results.csv'))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,'search_alg_results.csv'),mode='w', index=False,header=True)
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











