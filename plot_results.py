
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from preprocess_data2 import loadDatasetObj
from preprocess_data2 import RMSE,MAE,MAPE,get_Hzdata

# voltage, 460
# current, 52.5
def write_txt(txt,fname):
    f = open(fname, "w")
    f.write(txt)
    f.close()

results = '1s'
path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/pwr data paper 2/resampled data"
sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/results"
_,working_path = get_Hzdata(results,path,sav_path)
# working_path = os.path.join(base_path,results)

sav_path = os.path.join(working_path,'results_paper')
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

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
indeces_short = [alg_rename_short[f_i] for f_i in result_files]
indeces = [alg_rename[f_i] for f_i in result_files]
colors = ['r','g','c','y','k','b']

#%%
scale_mv = 460*52.5
fig, axs = plt.subplots(len(indeces),1, figsize=(11,11),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    axs[i].scatter(scale_mv*results_i['y_test'], scale_mv*results_i['y_test_pred'],alpha=0.3,s=15, color = colors[i],marker='o', linewidth=1.5)
    axs[i].set_title(indeces[i], fontsize=14, x=0.25, y=0.6)

    max_val_i = scale_mv*max(max(results_i['y_test']),max(results_i['y_test_pred']))
    X_line = np.arange(0,max_val_i,max_val_i/200)
    axs[i].plot(X_line,X_line)


    axs[i].set_ylabel('Predicted values', fontsize=9)
axs[i].set_xlabel('Actual values', fontsize=12)

fig.suptitle('Scatter plot of predictions vs real values for different algorithms\n for the active power (Watt)', fontsize=16, x=0.5, y=0.95)
plt.xticks( rotation=25 )

plt.savefig(os.path.join(sav_path,'scatter_plot_'+results+'.png'),bbox_inches='tight')

#%% bar plot RMSE
# patterns = [ "||" ,  "/",  "x","-","+",'//']
barWidth = 0.3
fig = plt.figure(figsize=(9,8),dpi=150)


data_res = []
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    RMSE_i = RMSE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    MAE_i = MAE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    MAPE_i = MAPE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    print([RMSE_i,MAE_i,MAPE_i])
    data_res.append([RMSE_i,MAE_i,MAPE_i])
    plt.bar(counter , RMSE_i , color=colors[counter], width=barWidth, edgecolor='white', label = indeces[counter])

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('RMSE (Watt)', fontweight='bold')

plt.xticks([r for r in range(len(indeces_short))], indeces_short)
_,b = plt.ylim()
plt.ylim([0,b*1.1])
# yticks_no = np.arange(0,lim,steps)
# yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
# plt.yticks(yticks_no,yticks)
plt.title('Bar plot of the RMSE for different algorithms')
# Create legend & Show graphic
plt.legend(prop={'size': 12},loc='best')
plt.gca().grid(True)
plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
plt.savefig(os.path.join(sav_path,'bar_plot_'+results+'.png'),bbox_inches='tight')

#%%
fig = plt.figure(figsize=(9,8),dpi=150)

for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    RMSE_i = RMSE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    MAE_i = MAE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    MAPE_i = MAPE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    plt.bar(counter , MAE_i , color=colors[counter], width=barWidth, edgecolor='white', label = indeces[counter])

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('MAE (Watt)', fontweight='bold')

plt.xticks([r for r in range(len(indeces_short))], indeces_short)
_,b = plt.ylim()
plt.ylim([0,b*1.1])
# yticks_no = np.arange(0,lim,steps)
# yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
# plt.yticks(yticks_no,yticks)
plt.title('Bar plot of the MAE for different algorithms')
# Create legend & Show graphic
plt.legend(prop={'size': 12},loc='best')
plt.gca().grid(True)
plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
plt.savefig(os.path.join(sav_path,'bar_plot_MAE'+results+'.png'),bbox_inches='tight')
#%%
fig = plt.figure(figsize=(9,8),dpi=150)

for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    RMSE_i = RMSE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    MAE_i = MAE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    MAPE_i = MAPE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
    plt.bar(counter , MAPE_i , color=colors[counter], width=barWidth, edgecolor='white', label = indeces[counter])

# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('MAPE (%)', fontweight='bold')

plt.xticks([r for r in range(len(indeces_short))], indeces_short)
_,b = plt.ylim()
plt.ylim([0,b*1.1])
# yticks_no = np.arange(0,lim,steps)
# yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
# plt.yticks(yticks_no,yticks)
plt.title('Bar plot of the MAPE for different algorithms')
# Create legend & Show graphic
plt.legend(prop={'size': 12},loc='best')
plt.gca().grid(True)
plt.show()
# if os.path.isfile(full_Acc_name):
#     os.remove(full_Acc_name)
plt.savefig(os.path.join(sav_path,'bar_plot_MAPE'+results+'.png'),bbox_inches='tight')


#%%
import pandas as pd
Metric = ['RMSE','MAE',"MAPE"]
data_res = np.array(data_res)
df = pd.DataFrame(data=data_res,columns=Metric,index = indeces)
print(df)
latex_txt = df.style.to_latex()


write_txt(latex_txt,os.path.join(sav_path,'results_table_latex.txt'))

# percentage_improvement = (data_res[:-1,:] - np.expand_dims(data_res[-1,:],axis=0) )/ (np.expand_dims(data_res[-1,:],axis=0))

#%% predictions vs real
# scale_mv = 460*52.5
fig, axs = plt.subplots(len(indeces),1, figsize=(11,11),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()
units = {'1s':'sec','5T':'5 min'}
for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    axs[i].plot(scale_mv*results_i['y_test'], color = 'red', linewidth=1.5, alpha = 0.6)
    axs[i].plot(scale_mv*results_i['y_test_pred'], color = 'blue', linewidth=0.8)
    axs[i].set_title(indeces[i], fontsize=14, x=0.5, y=0.6)
    axs[i].set_ylabel('Active power (Watt)', fontsize=8)
axs[i].set_xlabel('Timestamp ('+ units[results] +')' ,fontsize=10)


plt.legend(['Actual','Predicted'])

if results == '1s':
    f_title = 'Energy Prediction for '+str(int(len(results_i['y_test'])/3600)) +' hours ahead'
else:
    f_title = 'Energy Prediction for '+str(int(len(results_i['y_test'])/(12*24))) +' days ahead'


fig.suptitle(f_title, fontsize=16, x=0.5, y=0.9)
plt.xticks( rotation=25 )

plt.savefig(os.path.join(sav_path,'PredictionsVsReal'+results+'.png'),bbox_inches='tight')


#%%

result_files = os.listdir(sav_path)

result_files = ['Best_paramMod_FireflyAlgorithm.obj','Best_paramFireflyAlgorithm.obj']

alg_rename_itr = {'Best_paramMod_FireflyAlgorithm':'Modified Fire Fly',
              'Best_paramFireflyAlgorithm':'Fire Fly'}

full_file_path2 = [os.path.join(working_path,file) for file in result_files]

result_files = [file.split('.')[0] for file in result_files]

fig = plt.figure(figsize=(9,8),dpi=120)
markers = ['o-','o-']
data_hp = []
for counter,res_file in enumerate(full_file_path2):
    results_i = loadDatasetObj(res_file)
    data_hp.append(list(results_i['best_para_save'].values()))
    # max_val = max(max_val,max(max(results_i['y_test']),max(results_i['y_test_pred'])))
    plt.plot(results_i['a_itr'], scale_mv*np.sqrt(results_i['b_itr']),markers[counter],label=alg_rename_itr[result_files[counter]], linewidth=3.0)

plt.xlabel('Iteration', fontsize=13)
plt.ylabel('RMSE (Watt)', fontsize=13)
# plt.xlim(0)
plt.legend()
# plt.ylim(0)
plt.show()
plt.title('RMSE for the best firefly solution in each iteration vs iterations\n Convergence Graph')
plt.gca().grid(True)
plt.savefig(os.path.join(sav_path,'Conv_eval_comparison'+results+'.png'),bbox_inches='tight')




#%%
hp = list(results_i['best_para_save'].keys())
indeces_2 = [alg_rename_itr[f_i] for f_i in result_files]
df2 = pd.DataFrame(data=np.array(data_hp),columns=hp,index = indeces_2)
print(df2)


latex_txt_hp = df2.style.to_latex()

write_txt(latex_txt_hp,os.path.join(sav_path,'LSTM_HP_latex.txt'))
#%%


data_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/pwr data paper 2/resampled data\\1s.csv'
df = pd.read_csv(data_path)
df.set_index(pd.to_datetime(df.timestamp), inplace=True,drop=True,append=False)
# df = df.set_index('timestamp',drop=True,append=False)
# df.drop(columns=["timestamp"], inplace=True)
plt.figure(figsize=(10,7),dpi=180)
sns.heatmap(df.corr(), annot=True, annot_kws={"size": 18})
plt.savefig(os.path.join(sav_path,'corr_mat'+results+'.png'),bbox_inches='tight')

#%%
plt.figure(figsize=(10,7),dpi=180)
x_label = df.index[::int(len(df.index)/10)]
cols = ['P', 'Q', 'V', 'I']
# scale_mv = 460*52.5
df2=df.copy()
# df2['P'] = scale_mv * df2['P']
# df2['Q'] = scale_mv * df2['Q'] 
# df2['V'] = 460 * df2['V'] 
# df2['I'] = 52.5 * df2['I'] 
plt.plot(pd.to_datetime(df2.timestamp),df2[cols], linewidth=0.9)
# plt.xticks([r for r in range(len(x_label))], x_label)
cols_legend = ['Active power P (Watt)', 'Reactive power Q (VAR)', 'Voltage RMS V (Volt)', 'Current RMS I (Amp)']
plt.legend(cols_legend)
plt.xlabel('Time stamp')
plt.ylabel('Normalized values')
plt.title('Vizualization for the data')
plt.xticks(rotation=25)
plt.show()
plt.savefig(os.path.join(sav_path,'data_vis_'+results+'.png'),bbox_inches='tight')


