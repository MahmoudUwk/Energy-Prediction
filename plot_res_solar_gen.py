# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:59:22 2024

@author: mahmo
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pre_process_TFT import loadDatasetObj,RMSE,save_object,MAE,MAPE
import pandas as pd

# voltage, 460
# current, 52.5
def write_txt(txt,fname):
    f = open(fname, "w")
    f.write(txt)
    f.close()


working_path = 'results/1h'
sav_path = working_path+'/plots'

if not os.path.exists(sav_path):
    os.makedirs(sav_path)

result_files = os.listdir(working_path)

algorithms = ['LSTM','LR','SVR','RFR']

result_files = [file for file in result_files if file.endswith(".obj")]

# result_files = [file for c1,alg in enumerate(algorithms) for c2,file in enumerate(result_files) if alg in file]



alg_rename = {'LSTM':'LSTM',
              'LR':'LR',
              'SVR':'SVR',
              'RFR':'RFR'}

full_file_path = [os.path.join(working_path,file) for file in result_files]

indeces = [algorithms[c2] for c1,file in enumerate(result_files) for c2,alg in enumerate(algorithms) if alg in file]



# full_file_path = [file for file in full_file_path if os.path.exists(file)]

# result_files = [file.split('.')[0] for file in result_files]
# indeces_short = [alg_rename_short[f_i] for f_i in result_files]
indeces = [alg_rename[f_i] for f_i in indeces]
indeces_short =indeces
colors = ['#047495','#632de9','#a4be5c','#acfffc']


#%%
#scatter plots
fig, axs = plt.subplots(2,int(len(result_files)/2), figsize=(20,8),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    a1 = results_i['y_test']
    a2 = results_i['y_test_pred']
    axs[i].scatter(a1,a2 ,alpha=0.5,s=15, color = colors[i],marker='o', linewidth=1.5)
    axs[i].set_title(indeces[i], fontsize=14, x=0.25, y=0.6)

    max_val_i = max(max(a1),max(a2))
    X_line = np.arange(0,max_val_i,max_val_i/200)
    axs[i].plot(X_line,X_line)
    axs[i].set_xlabel('Actual values', fontsize=12)

    if i == 2 or i == 0:
        axs[i].set_ylabel('Predicted values', fontsize=10)


fig.suptitle('Scatter plot of predictions vs real values \n for solar energy generation prediction (Kw)', fontsize=12, x=0.5, y=0.95)
# plt.xticks( rotation=25 )

plt.savefig(os.path.join(sav_path,'scatter_plot.png'),bbox_inches='tight')

#%% bar plot RMSE
# patterns = [ "||" ,  "/",  "x","-","+",'//']
from sklearn.metrics import r2_score
barWidth = 0.3
# fig = plt.figure(figsize=(14,8),dpi=150)
fig, axs = plt.subplots(2,int(len(result_files)/2), figsize=(20,8),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
axs = axs.ravel()

data_res = []
r2 = []
train_time_all = []
test_time_all = []
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    a1 = results_i['y_test']
    a2 = results_i['y_test_pred']
    train_time_all.append(results_i['train_time'])
    test_time_all.append(results_i['test_time'])
    
    RMSE_i = RMSE(a1,a2)
    MAE_i = MAE(a1,a2)
    MAPE_i = MAPE(a1,a2)
    r2_i = r2_score(a1,a2)
    data_res.append([RMSE_i,MAE_i,MAPE_i,r2_i])
    

    axs[0].bar(counter , RMSE_i , color=colors[counter], width=barWidth, hatch="x", edgecolor='black', label = indeces[counter])
    axs[1].bar(counter , MAE_i , color=colors[counter], width=barWidth, hatch="x", edgecolor='black', label = indeces[counter])
    axs[2].bar(counter , MAPE_i , color=colors[counter], width=barWidth, hatch="x", edgecolor='black', label = indeces[counter])
    axs[3].bar(counter , r2_i , color=colors[counter], width=barWidth, hatch="x", edgecolor='black', label = indeces[counter])

metric_bar = ['RMSE (kW)','MAE (kW)',"MAPE(%)","R square"]
for c in range(4):
    axs[c].set_xlabel('Algorithm', fontweight='bold')
    axs[c].set_ylabel(metric_bar[c], fontweight='bold')
    axs[c].set_xticks([r for r in range(len(indeces_short))], indeces_short)
    axs[c].grid(True)
    
    
fig.suptitle('Bar plot for the evaluation metrics', fontsize=12, x=0.5, y=0.92)

# # Add xticks on the middle of the group bars
# plt.xlabel('Algorithm', fontweight='bold')
# plt.ylabel('RMSE', fontweight='bold')
# plt.xticks([r for r in range(len(indeces_short))], indeces_short)
# _,b = plt.ylim()
# plt.ylim([0,b*1.1])
# plt.title('Bar plot of the RMSE for different algorithms')
# # Create legend & Show graphic
# fig.legend(prop={'size': 12},loc=(0.08,0.75))

# plt.show()
plt.savefig(os.path.join(sav_path,'bar_plot.png'),bbox_inches='tight')
#%% train and test times
Metric = ['RMSE (kW)','MAE (kW)',"MAPE(%)","R square",'Train time','Test time']
data_res = np.array(data_res)

def process_time(train_time_all):
    train_time_all2 = []
    for TT in train_time_all:
        if isinstance(TT, list):
            TT = sum(TT)
        if TT < 1 :
            TT = TT*60
            TT = str(np.round(TT,2))+' s'
        else:
            if TT < 60:
                TT = str(np.round(TT,2))+' min'
            else:
                TT = TT / 60
                TT = str(np.round(TT,2))+' hr'
        train_time_all2.append(TT)
    return train_time_all2

data_res = np.round(np.array(data_res),3)
TT = np.array([process_time(train_time_all),process_time(test_time_all)]).T
dat = np.concatenate((data_res, TT),axis=1)
df = pd.DataFrame(data=dat,columns=Metric,index = indeces)
print(df)
latex_txt = df.style.to_latex()


write_txt(latex_txt,os.path.join(sav_path,'results_table_latex.txt'))
#%% predictions vs real
# scale_mv = 460*52.5
fig, axs = plt.subplots(len(indeces),1, figsize=(20,11),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

# M_id = 880
#
append_test = []
for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)


    axs[i].plot(np.squeeze(results_i['y_test']),  color = 'red', linewidth=1, alpha = 0.6)
    axs[i].plot(np.squeeze(results_i['y_test_pred']),  color = 'blue', linewidth=0.5)
    axs[i].set_title(indeces[i], fontsize=10, x=0.55, y=0.6)
    # axs[i].set_xticks(np.arange(0,len_i,10) , 5*np.arange(0,len_i,10) )
    axs[i].set_ylabel('Prediction (kW)', fontsize=8)
    

axs[i].set_xlabel('Timestamp (hour)' ,fontsize=10)

#np.mean(np.array(append_test),axis=1)
# plt.legend(['Actual','Predicted'])
fig.legend(['Actual','Predicted'],prop={'size': 12},loc=(0.8,0.89))


fig.suptitle('Prediction vs true values for solar energy generation prediction', fontsize=12, x=0.5, y=0.92)
# plt.xticks( np.arange(0,5,len_i*5) )

plt.savefig(os.path.join(sav_path,'PredictionsVsReal.png'),bbox_inches='tight')
# plt.savefig(os.path.join(sav_path,'PredictionsVsReal.png'),bbox_inches='tight',format='eps')