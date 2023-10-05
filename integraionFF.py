# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:20:40 2023

@author: mahmo
"""

from sklearn_nature_inspired_algorithms.model_selection import NatureInspiredSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVR
from niapy.algorithms.basic import Mod_FireflyAlgorithm
from niapy.algorithms.basic import FireflyAlgorithm
from lssvr import LSSVR
# from sklearn.datasets import fetch_openml

from sklearn.datasets import load_diabetes
X_train,y_train = load_diabetes(return_X_y=True)




#%%
algorithm = Mod_FireflyAlgorithm.Mod_FireflyAlgorithm()
# algorithm = FireflyAlgorithm()
algorithm.set_parameters()


param_grid = { 
    'C': [10**c for c in range(-3,5)], 
    'gamma': [10**g for g in range(-3,5)],
}

clf = LSSVR(kernel='rbf')

nia_search = NatureInspiredSearchCV(
    clf,
    param_grid,
    algorithm=algorithm, # hybrid bat algorithm
    population_size=50,
    max_n_gen=25,
    max_stagnating_gen=10,
    runs=3,
    random_state=None, # or any number if you want same results on each run
)

nia_search.fit(X_train, y_train)

# the best params are stored in nia_search.best_params_
# finally you can train your model with best params from nia search
print("best parameters",nia_search.best_params_)
new_clf = LSSVR(**nia_search.best_params_)