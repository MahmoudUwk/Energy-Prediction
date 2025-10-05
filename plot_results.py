
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

from config import EXPECTED_RESULT_FILES, DEFAULT_RESULTS_DATASET
from tools.preprocess_data2 import (
    RMSE,
    MAE,
    MAPE,
    get_SAMFOR_data,
    loadDatasetObj,
)

# voltage, 460
# current, 52.5
def write_txt(txt,fname):
    f = open(fname, "w")
    f.write(txt)
    f.close()

results = DEFAULT_RESULTS_DATASET


if results == 'Home':
    scale_mv = 1
else:
    scale_mv = 460 * 52.5
working_path = get_SAMFOR_data(0, results, 0, 1)
# working_path = os.path.join(base_path,results)

sav_path = os.path.join(working_path,'results_paper')
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

result_files = os.listdir(working_path)

result_files = EXPECTED_RESULT_FILES

# result_files = [file for file in result_files if file.endswith(".obj") and file in algorithms]

alg_rename = {'FireflyAlgorithm':'LSTM FF',
              'Mod_FireflyAlgorithm':'LSTM Modified FF (Proposed)',
              'LSTM':'LSTM without HP tuning',
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

# full_file_path = [file for file in full_file_path if os.path.exists(file)]

result_files = [file.split('.')[0] for file in result_files]
indeces_short = [alg_rename_short[f_i] for f_i in result_files]
indeces = [alg_rename[f_i] for f_i in result_files]
# colors = ['r','g','c','y','k','b']
separation = 30
colors = sns.color_palette("magma",6*separation).as_hex()[::separation] #cubehelix magma Spectral
barWidth = 0.3
linewidth=1.5
Metric = ['RMSE (W)','MAE (W)',"MAPE (%)","R square score"]
Metric_name = ['RMSE','MAE',"MAPE","r2_score"]
fromatt = 'eps'# 'eps' or 'png'
fs = 17
# scatter plot
fig, axs = plt.subplots(int(len(result_files)/2),2, figsize=(15,8),dpi=150, facecolor='w', edgecolor='k')
# fig, axs = plt.subplots(len(indeces),1, figsize=(11,11),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.1)

axs = axs.ravel()

for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    axs[i].scatter(scale_mv*results_i['y_test'], scale_mv*results_i['y_test_pred'],alpha=0.5,s=15, color = colors[i],marker='o', linewidth=1.5)
    axs[i].set_title(indeces[i], fontsize=14, x=0.27, y=0.62)

    max_val_i = scale_mv*max(max(results_i['y_test']),max(results_i['y_test_pred']))
    X_line = np.arange(0,max_val_i,max_val_i/200)
    axs[i].plot(X_line,X_line)


    if i in [0,2,4]:
        axs[i].set_ylabel('Predicted values', fontsize=fs)
    axs[i].set_xlabel('Actual values', fontsize=fs)

fig.suptitle('Scatter plot of predictions vs real values for different algorithms for the active power (Watt)', fontsize=fs, x=0.5, y=0.95)
plt.xticks( rotation=25 )

# plt.savefig(os.path.join(sav_path,'scatter_plot_'+results+'.png'),bbox_inches='tight',format='png')


def plot_bar(results,barWidth,linewidth,full_file_path,ind_plot,Metric,Metric_name):

    fig = plt.figure(figsize=(14,8),dpi=150)

    data_res = []
    for counter,res_file in enumerate(full_file_path):
        results_i = loadDatasetObj(res_file)
        RMSE_i = RMSE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
        MAE_i = MAE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
        MAPE_i = MAPE(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
        r2_score_i = r2_score(scale_mv*results_i['y_test'],scale_mv*results_i['y_test_pred'])
        # print([RMSE_i,MAE_i,MAPE_i])
        row = [RMSE_i,MAE_i,MAPE_i,r2_score_i]
        data_res.append(row)
        plt.bar(counter , row[ind_plot] , color=colors[counter],linewidth=linewidth, width=barWidth, hatch="xxxx", edgecolor=colors[counter], label = indeces[counter],fill=False)


    # Add xticks on the middle of the group bars
    plt.xlabel('Algorithm', fontweight='bold', fontsize=fs)
    plt.ylabel(Metric[ind_plot], fontweight='bold', fontsize=fs)

    plt.xticks([r for r in range(len(indeces_short))], indeces_short)
    _,b = plt.ylim()
    plt.ylim([0,b*1.1])
    # yticks_no = np.arange(0,lim,steps)
    # yticks = [ str((100*n).round(3))+' %' for n in yticks_no]
    # plt.yticks(yticks_no,yticks)
    plt.title('Bar plot of the '+Metric[ind_plot] + ' for different algorithms', fontsize=fs)
    # Create legend & Show graphic
    plt.legend(prop={'size': fs},loc='best')
    plt.gca().grid(True)
    plt.show()
    # if os.path.isfile(full_Acc_name):
    #     os.remove(full_Acc_name)
    # plt.savefig(os.path.join(sav_path,'bar_plot_'+results+Metric_name[ind_plot]+'.'+fromatt),bbox_inches='tight',format=fromatt)
    return data_res

data_res = [plot_bar(results,barWidth,linewidth,full_file_path,i,Metric,Metric_name) for i in range(4)][0]


#%%
import pandas as pd

data_res = np.array(data_res)
df = pd.DataFrame(data=data_res,columns=Metric,index = indeces).round(4)
print(df)
latex_txt = df.style.to_latex()


write_txt(latex_txt,os.path.join(sav_path,'results_table_latex.txt'))

# percentage_improvement = (data_res[:-1,:] - np.expand_dims(data_res[-1,:],axis=0) )/ (np.expand_dims(data_res[-1,:],axis=0))

#%% predictions vs real
# scale_mv = 460*52.5
fig, axs = plt.subplots(len(indeces),1, figsize=(20,11),dpi=150, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()
units = {'1s':'sec','5T':'5 min'}
for i,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    axs[i].plot(scale_mv*np.squeeze(results_i['y_test']), color = 'red', linewidth=1.5, alpha = 0.6)
    axs[i].plot(scale_mv*np.squeeze(results_i['y_test_pred']), color = 'blue', linewidth=0.8)
    axs[i].set_title(indeces[i], fontsize=14, x=0.5, y=0.6)
    axs[i].set_ylabel('Active power (Watt)', fontsize=8)
axs[i].set_xlabel('Timestamp ('+ units[results] +')' ,fontsize=fs)


plt.legend(['Actual','Predicted'])

if results == '1s':
    f_title = 'Energy Prediction for '+str(int(len(results_i['y_test'])/3600)) +' hours ahead'
else:
    f_title = 'Energy Prediction for '+str(int(len(results_i['y_test'])/(12*24))) +' days ahead'


fig.suptitle(f_title, fontsize=fs, x=0.5, y=0.92)
# plt.xticks( rotation=25 )

# plt.savefig(os.path.join(sav_path,'PredictionsVsReal'+results+'.'+fromatt),bbox_inches='tight',format=fromatt)


#%%conv graph

result_files = os.listdir(sav_path)

result_files = ['Best_paramFireflyAlgorithm.obj','Best_paramMod_FireflyAlgorithm.obj']

alg_rename_itr = {'Best_paramMod_FireflyAlgorithm':'Modified Fire Fly',
              'Best_paramFireflyAlgorithm':'Fire Fly'}

full_file_path2 = [os.path.join(working_path,file) for file in result_files]

result_files = [file.split('.')[0] for file in result_files]

fig = plt.figure(figsize=(14,8),dpi=150)
markers = ['o-','o-']
fillstyles = ['top','bottom']
data_hp = []
for counter,res_file in enumerate(full_file_path2):
    results_i = loadDatasetObj(res_file)
    data_hp.append(list(results_i['best_para_save'].values()))
    # max_val = max(max_val,max(max(results_i['y_test']),max(results_i['y_test_pred'])))
    plt.step(results_i['a_itr'], scale_mv*np.sqrt(results_i['b_itr']),markers[counter],markersize=10,dash_joinstyle='bevel'
             ,fillstyle=fillstyles[counter],color=colors[counter+4],label=alg_rename_itr[result_files[counter]], linewidth=3)

plt.xlabel('Iteration', fontsize=fs)
plt.ylabel('RMSE (Watt)', fontsize=fs)
plt.xticks(results_i['a_itr'], range(1,len(results_i['a_itr'])+1))
# plt.xlim(0)
plt.legend(prop={'size': fs},loc='best')
# plt.ylim(0)
plt.show()
plt.title('Convergence Graph for the best firefly solution in each iteration vs the iteration number', fontsize=fs)
plt.gca().grid(True)
# plt.savefig(os.path.join(sav_path,'Conv_eval_comparison'+results+'.'+fromatt),bbox_inches='tight',format=fromatt)




#%%
hp = list(results_i['best_para_save'].keys())
indeces_2 = [alg_rename_itr[f_i] for f_i in result_files]
df2 = pd.DataFrame(data=np.array(data_hp),columns=hp,index = indeces_2)
df2[hp[:3]] = df2[hp[:3]].astype('int')
print(df2)


latex_txt_hp = df2.style.to_latex()

# write_txt(latex_txt_hp,os.path.join(sav_path,'LSTM_HP_latex.txt'))
#%%


# data_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Energy Prediction Project/pwr data paper 2/resampled data//1s.csv'
# df = pd.read_csv(data_path)
# df.set_index(pd.to_datetime(df.timestamp), inplace=True,drop=True,append=False)
# # df = df.set_index('timestamp',drop=True,append=False)
# # df.drop(columns=["timestamp"], inplace=True)
# plt.figure(figsize=(10,7),dpi=180)
# sns.heatmap(df.corr(), annot=True, annot_kws={"size": 18})
# plt.savefig(os.path.join(sav_path,'corr_mat'+results+'.png'),bbox_inches='tight')

# #%%
# plt.figure(figsize=(10,7),dpi=180)
# x_label = df.index[::int(len(df.index)/10)]
# cols = ['P', 'Q', 'V', 'I']
# # scale_mv = 460*52.5
# df2=df.copy()
# # df2['P'] = scale_mv * df2['P']
# # df2['Q'] = scale_mv * df2['Q'] 
# # df2['V'] = 460 * df2['V'] 
# # df2['I'] = 52.5 * df2['I'] 
# plt.plot(pd.to_datetime(df2.timestamp),df2[cols], linewidth=0.9)
# # plt.xticks([r for r in range(len(x_label))], x_label)
# cols_legend = ['Active power P (Watt)', 'Reactive power Q (VAR)', 'Voltage RMS V (Volt)', 'Current RMS I (Amp)']
# plt.legend(cols_legend)
# plt.xlabel('Time stamp')
# plt.ylabel('Normalized values')
# plt.title('Vizualization for the data')
# plt.xticks(rotation=25)
# plt.show()
# plt.savefig(os.path.join(sav_path,'data_vis_'+results+'.png'),bbox_inches='tight')
#%%
abs_erros = []
for counter,res_file in enumerate(full_file_path):
    results_i = loadDatasetObj(res_file)
    a_true = scale_mv*results_i['y_test']
    a_pred = np.squeeze(scale_mv*results_i['y_test_pred'])
    abs_erros.append(np.abs(a_true-a_pred))

plt.figure(figsize=(14,8))
# plt.boxplot(abs_erros)
indeces2 = ['SVR',
 'RFR',
 'SAMFOR',
 'LSTM',
 'LSTM FF',
 'LSTM Mod FF']
boxplot_data  = plt.boxplot(abs_erros, labels=indeces2, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='teal'),
            capprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='teal', linewidth=2),
            flierprops=dict(markerfacecolor='r', marker='D'),
            medianprops=dict(color='green', linewidth=1.5),
             showfliers=False
            )
# plt.title(title, fontsize=16)
plt.ylabel("Absolute Error", fontsize=fs)
plt.grid()
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(False)
plt.savefig(os.path.join(sav_path,'boxplot.eps'),bbox_inches='tight', format='eps')


relevant_data = []
for i in range(len(abs_erros)):
    # Extract Q1 and Q3 from the box vertices
    box_path = boxplot_data['boxes'][i].get_path()
    vertices = box_path.vertices
    q1 = vertices[0, 1]  # Y-coordinate of the first corner (Q1)
    q3 = vertices[2, 1]  # Y-coordinate of the third corner (Q3)

    # Extract median, whiskers
    median = boxplot_data['medians'][i].get_ydata()[0]
    whisker_low = boxplot_data['whiskers'][2 * i].get_ydata()[1]
    whisker_high = boxplot_data['whiskers'][2 * i + 1].get_ydata()[1]

    # Collect data
    data = {
        "label": indeces[i],
        "median": median,
        "q1": q1,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }
    relevant_data.append(data)

#%%
from scipy.stats import f_oneway

F_stat, p_value = f_oneway(*abs_erros)


