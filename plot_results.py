# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:29:48 2024

@author: mahmo
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from preprocess_data import loadDatasetObj
from preprocess_data import RMSE,MAE,MAPE

base_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/results"

results = '5T'

working_path = os.path.join(base_path,results)

result_files = os.listdir(working_path)



result_files = [file for file in result_files if file.endswith(".obj")]

full_file_path = [os.path.join(working_path,file) for file in result_files]

result_files = [file.split('.')[0] for file in result_files]
fig = plt.figure(figsize=(9,8),dpi=120)
max_val = 0
markers = ["o","P","*","x",'+']
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    max_val = max(max_val,max(max(results_i['y_test']),max(results_i['y_test_pred'])))
    plt.scatter(results_i['y_test'], results_i['y_test_pred'],s=15 ,alpha=0.3,marker=markers[counter],label=result_files[counter])

max_val = max_val
X_line = np.arange(0,max_val,max_val/200)
plt.plot(X_line,X_line)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.xlim(0)
plt.legend()
plt.ylim(0)
plt.show()

#%%

# lim = lim_up * round(100*np.max(error_rate)/lim_up)/100

# set height of bar
colors = ['r','g','b','y','k']
patterns = [ "\\" ,  "/",  "x","-","+"]
barWidth = 0.15
fig = plt.figure(figsize=(9,8),dpi=120)
 
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    RMSE_i = RMSE(results_i['y_test'],results_i['y_test_pred'])
    MAE_i = MAE(results_i['y_test'],results_i['y_test_pred'])
    MAPE_i = MAPE(results_i['y_test'],results_i['y_test_pred'])
    plt.bar(counter , RMSE_i , color=colors[counter], width=barWidth, edgecolor='white', label = result_files[counter], hatch=patterns[counter])

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('RMSE', fontweight='bold')
plt.xticks([r for r in range(len(result_files))], result_files)
_,b = plt.ylim()
plt.ylim([0,b*1.1])
# yticks_no = np.arange(0,lim,steps)
# yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
# plt.yticks(yticks_no,yticks)
 
# Create legend & Show graphic
plt.legend(prop={'size': 10},loc='best')
plt.gca().yaxis.grid(True)
#plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
# plt.savefig(full_Acc_name,bbox_inches='tight')
# plt.close()




















