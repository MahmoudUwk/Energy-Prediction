from sklearn_nature_inspired_algorithms.model_selection import NatureInspiredSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVR
from niapy.algorithms.basic import Mod_FireflyAlgorithm
from niapy.algorithms.basic import *
# from lssvr import LSSVR
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import os
from preprocess_data import RMSE,MAE,MAPE,get_SAMFOR_data

save_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/results/Models'
option = 1
X_train,y_train,X_test,y_test = get_SAMFOR_data(option)
print(X_train.shape,X_test.shape)
#%%
n_pop = 25
itr = 10
save_name = 'search_alg_LSSVR'+'pop'+str(n_pop)+'itr'+str(itr)+'.csv'
algorithm = Mod_FireflyAlgorithm.Mod_FireflyAlgorithm()
# algorithm = FireflyAlgorithm()

algorithms = [BeesAlgorithm(),Mod_FireflyAlgorithm.Mod_FireflyAlgorithm(),FireflyAlgorithm(), ArtificialBeeColonyAlgorithm(),BatAlgorithm()
              ,BacterialForagingOptimization(),ClonalSelectionAlgorithm(),
              CuckooSearch(),CatSwarmOptimization(),ForestOptimizationAlgorithm(),
              FlowerPollinationAlgorithm(),BareBonesFireworksAlgorithm(),
              GravitationalSearchAlgorithm(),GlowwormSwarmOptimization(),GreyWolfOptimizer(),HarrisHawksOptimization(),
              HarmonySearch(),KrillHerd(),MonarchButterflyOptimization(),MothFlameOptimizer(),ParticleSwarmAlgorithm()
             , SineCosineAlgorithm()]

#%%
param_grid = { 
    'C': [10**c for c in range(-3,3)], 
    'epsilon': [10**g for g in range(-3,3)]
    #'gamma': [10**g for g in range(-3,3)],
}
#%%
cols = ["Algorithm", "RMSE", "MAE", "MAPE", "time (s)","C","gamma"]
df3 = pd.DataFrame(columns=cols)
if not os.path.isfile(os.path.join(save_path,save_name)):
    df3.to_csv(os.path.join(save_path,save_name),index=False)
    #%%

for alg in range(len(algorithms)):
 
    clf = SVR(kernel='rbf')
    
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
    nia_search.fit(X_train, np.squeeze(y_train))
    end_time = time.time()
    time_passed = end_time - start_time
    # the best params are stored in nia_search.best_params_
    # finally you can train your model with best params from nia search
    #%%

    print("best parameters",nia_search.best_params_)
    new_clf = SVR(**nia_search.best_params_)
    new_clf.fit(X_train, np.squeeze(y_train))
    
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
    df = pd.read_csv(os.path.join(save_path,save_name))
    df.loc[len(df)] = row
    print(df)
    df.to_csv(os.path.join(save_path,save_name),mode='w', index=False,header=True)
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











