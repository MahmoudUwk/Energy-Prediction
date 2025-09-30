# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:34:22 2024

@author: Admin
"""
import pickle
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

path1 = 'C:/Users/Admin/Desktop/New folder/results/1s/Best_paramFireflyAlgorithm.obj'
path2 = 'C:/Users/Admin/Desktop/New folder/results/1s/Best_paramMod_FireflyAlgorithm.obj'

p1 = loadDatasetObj(path1)
print(p1)
p2 = loadDatasetObj(path2)
print(p2)