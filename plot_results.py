
import os
import matplotlib.pyplot as plt
import numpy as np
from preprocess_data2 import loadDatasetObj
from preprocess_data2 import RMSE,MAE,MAPE,get_Hzdata



results = '1s'
_,working_path = get_Hzdata(results)
# working_path = os.path.join(base_path,results)

result_files = os.listdir(working_path)

result_files = ['SVR.obj','RFR.obj','SAMFOR.obj','LSTM.obj','FireflyAlgorithm.obj','Mod_FireflyAlgorithm.obj']

# result_files = [file for file in result_files if file.endswith(".obj") and file in algorithms]

alg_rename = {'FireflyAlgorithm':'LSTM FF',
              'Mod_FireflyAlgorithm':'LSTM Modified FF (Proposed)',
              'LSTM':'LSTM without hyper-parameter tuning',
              'RFR':'RFR',
              'SAMFOR':'SAMFOR',
              'SVR':'SVR'}

alg_rename_short = {'FireflyAlgorithm':'LSTM FF',
              'Mod_FireflyAlgorithm':'Proposed',
              'LSTM':'LSTM',
              'RFR':'RFR',
              'SAMFOR':'SAMFOR',
              'SVR':'SVR'}

full_file_path = [os.path.join(working_path,file) for file in result_files]

result_files = [file.split('.')[0] for file in result_files]

#%% scatter plot
fig = plt.figure(figsize=(9,8),dpi=120)
max_val = 0
markers = ["o","P","*","x",'+','<']
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
plt.title('Scatter plot of predictions vs real values for different algorithms')
plt.savefig(os.path.join(working_path,'scatter_plot_'+results+'.png'))
#%%

# lim = lim_up * round(100*np.max(error_rate)/lim_up)/100

# set height of bar
colors = ['r','g','c','y','k','b']
patterns = [ "||" ,  "/",  "x","-","+",'//']
barWidth = 0.15
fig = plt.figure(figsize=(9,8),dpi=120)

indeces_short = [alg_rename_short[f_i] for f_i in result_files]
indeces = [alg_rename[f_i] for f_i in result_files]
data_res = []
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    RMSE_i = RMSE(results_i['y_test'],results_i['y_test_pred'])
    MAE_i = MAE(results_i['y_test'],results_i['y_test_pred'])
    MAPE_i = MAPE(results_i['y_test'],results_i['y_test_pred'])
    data_res.append([RMSE_i,MAE_i,MAPE_i])
    plt.bar(counter , RMSE_i , color=colors[counter], width=barWidth, edgecolor='white', label = indeces[counter], hatch=patterns[counter])

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('RMSE', fontweight='bold')

plt.xticks([r for r in range(len(indeces_short))], indeces_short)
_,b = plt.ylim()
plt.ylim([0,b*1.1])
# yticks_no = np.arange(0,lim,steps)
# yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
# plt.yticks(yticks_no,yticks)
plt.title('Bar plot of the RMSE for different algorithms')
# Create legend & Show graphic
plt.legend(prop={'size': 12},loc='best')
plt.gca().yaxis.grid(True)
plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
plt.savefig('bar_plot_'+results+'.png',bbox_inches='tight')
# plt.close()


#%%
import pandas as pd
Metric = ['RMSE','MAE',"MAPE"]
indeces = [alg_rename[f_i] for f_i in result_files]
df = pd.DataFrame(data=np.array(data_res),columns=Metric,index = indeces)
print(df)


latex_txt = df.style.to_latex()

#%%

max_val = 0
markers = ["o","P","*","x",'+','<']
scale_mv = 1000
# fig = plt.figure(figsize=(9,8),dpi=120)
fig, axs = plt.subplots(len(indeces),1, figsize=(10,16),dpi=100, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    max_val = max(max_val,max(max(results_i['y_test']),max(results_i['y_test_pred'])))
    axs[i].plot(scale_mv*results_i['y_test'], color = 'red', linewidth=2.0, alpha = 0.6)
    axs[i].plot(scale_mv*results_i['y_test_pred'], color = 'blue', linewidth=0.8)
    axs[i].set_title(indeces[i], fontsize=9)


plt.legend(['Actual','Predicted'])
# plt.legend(prop={'size': 12},loc='best')
plt.xlabel('Timestamp')
plt.ylabel('mW')

fig.suptitle('Energy Prediction '+str(int(len(results_i['y_test'])/3600)) +' hours ahead', fontsize=16)
plt.xticks( rotation=25 )

plt.savefig(os.path.join(working_path,'PredictionsVsReal'+results+'.png'),bbox_inches='tight')


#%%

result_files = os.listdir(working_path)

result_files = ['Best_paramMod_FireflyAlgorithm.obj','Best_paramFireflyAlgorithm.obj']

alg_rename = {'Best_paramMod_FireflyAlgorithm':'Modified Fire Fly',
              'Best_paramFireflyAlgorithm':'Fire Fly'}

full_file_path = [os.path.join(working_path,file) for file in result_files]

result_files = [file.split('.')[0] for file in result_files]

#convergence comparison
fig = plt.figure(figsize=(9,8),dpi=120)
markers = ['o-','.-']
data_hp = []
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    data_hp.append(list(results_i['best_para_save'].values()))
    # max_val = max(max_val,max(max(results_i['y_test']),max(results_i['y_test_pred'])))
    plt.plot(results_i['a_itr'], results_i['b_itr'],markers[counter],label=alg_rename[result_files[counter]])



plt.xlabel('Iteration')
plt.ylabel('MSE')
# plt.xlim(0)
plt.legend()
# plt.ylim(0)
plt.show()
plt.title('Convergence Graph')
plt.gca().grid(True)
plt.savefig(os.path.join(working_path,'Conv_eval_comparison'+results+'.png'))


hp = list(results_i['best_para_save'].keys())
indeces = [alg_rename[f_i] for f_i in result_files]
df = pd.DataFrame(data=np.array(data_hp),columns=hp,index = indeces)
print(df)


latex_txt_hp = df.style.to_latex()