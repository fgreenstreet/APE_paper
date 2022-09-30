import os
import pandas as pd
import seaborn as sns
import sys
import numpy as np
from dlc_processing_utils import *
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
from psychometric_utils import get_all_psychometric_session_dlc, norm_data_for_param
from freely_moving_photometry_analysis.utils.regression.regression_plotting_utils import make_box_plot
import matplotlib.pyplot as plt
from plotting import *
import statsmodels.api as sm
import scipy.stats
from set_global_params import post_processed_tracking_data_path

# this came from a different repo so check imports
#   TODO: sort out imports
#   TODO: remove tracking analysis from this
mouse_ids = ['SNL_photo57']
#mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
site = 'tail'
save = False
num_sessions = 4
key = 'fitted max cumsum ang vel'
alignment='choice'
all_trial_data = get_all_psychometric_session_dlc(mouse_ids, site, save=False, load_saved=False, key=key, get_movement=False, align_to=alignment, num_sessions=num_sessions)
#all_trial_data.to_csv("all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces.csv".format(site))
#all_trial_data.to_pickle("all_trial_data_{}_{}_contra_ipsi_last_trial_confidence_and_traces_{}_aligned.pkl".format(mouse_ids[0], site, alignment))
all_trial_data.to_pickle("all_trial_data_{}_{}_SNL_photo57.pkl".format( site, alignment))
pvals_m = []
slopes_m = []
r_squared_m = []
pvals_s = []
slopes_s = []
r_squared_s = []
for mouse in mouse_ids:
    plt.figure()
    mouse_data = all_trial_data[all_trial_data['mouse'] == mouse]
    mouse_data = mouse_data.dropna(axis=0)
    Y = mouse_data['norm APE'].reset_index(drop=True)
    X = sm.add_constant(np.abs(mouse_data[key]).reset_index(drop=True))
    plt.scatter(X[key], Y)
    #X = pd.concat([np.abs(mouse_data[key]),mouse_data['trial type']], axis=1).reset_index(drop=True)
    model = sm.OLS(Y, X)
    results = model.fit()
    dopamine_estimate = results.predict(X)
    residual = Y - dopamine_estimate
    new_X = sm.add_constant(mouse_data['trial type'].reset_index(drop=True))
    new_model = sm.OLS(residual, new_X)
    new_results = new_model.fit()
    pvals_m.append(results.pvalues)
    slopes_m.append(results.params)
    r_squared_m.append(results.rsquared)
    pvals_s.append(new_results.pvalues)
    slopes_s.append(new_results.params)
    r_squared_s.append(new_results.rsquared)

betas_movement = [b[key] for b in slopes_m]
betas_stimulus_intensity = [b['trial type'] for b in slopes_s]
betas_df = pd.concat(slopes_m + slopes_s).reset_index().rename({'index':'parameter',0: 'regression\ncoefficient'}, axis='columns')
betas_df = betas_df.replace({key: 'max cumulative\nangular velocity'})
font = {'size': 7}
matplotlib.rc('font', **font)
fig, axs = plt.subplots(2, 3, figsize=[6, 4])
make_box_plot(betas_df, axs[1, 2], dx='parameter', dy='regression\ncoefficient', pal='pastel_d', set_ylims=False)
axs[1, 1].scatter(new_X['trial type'], residual, s=2, color='grey')
axs[1, 0].scatter(new_X['trial type'], Y, s=2, color='grey')
plt.tight_layout()
print('pval movement:', scipy.stats.ttest_1samp(betas_movement, 0.0)[1])
print('pval stimulus intensity:', scipy.stats.ttest_1samp(betas_stimulus_intensity, 0.0)[1])
print('pval related ttest:', scipy.stats.ttest_rel(betas_stimulus_intensity, betas_movement)[1])



q_data = all_trial_data[all_trial_data['recording site'] == site]
s_data = all_trial_data[all_trial_data['recording site'] == 'shuffled '+ site]
print('a')
example_mouse = 'SNL_photo26'
example_date = '20200810'
save_out_folder = post_processed_tracking_data_path + example_mouse
if not os.path.exists(save_out_folder):
    os.makedirs(save_out_folder)
movement_param_file = os.path.join(save_out_folder, 'contra_APE_tracking{}_{}_psychometric.pkl'.format(example_mouse, example_date))
if os.path.isfile(movement_param_file):
    quantile_data = pd.read_pickle(movement_param_file)

fig, axs = plt.subplots(1,3, figsize=(12,4))
key='max angular velocity'
norm_cumsum_data = norm_data_for_param(all_trial_data, experiment_record, trial_types, site='tail', key=key)
for i in trial_types[:-1]:
    trial_type_data = norm_cumsum_data.loc[all_trial_data['trial type'] == i]
    print(trial_type_data.shape)
    axs[0].scatter(i, trial_type_data['norm APE'].mean()
                   , color=colours[int(i - 1)])
for i in trial_types[:-1]:
    trial_type_data = norm_cumsum_data.loc[norm_cumsum_data['trial type'] == i]
    print(trial_type_data.shape)
    axs[1].scatter(i, trial_type_data[key].mean()
                   , color=colours[int(i - 1)])
for i in trial_types[:-1]:
    trial_type_data = norm_cumsum_data.loc[norm_cumsum_data['trial type'] == i]
    print(trial_type_data.shape)
    axs[2].scatter(trial_type_data[key].mean(), trial_type_data['norm APE'].mean()
                   , color=colours[int(i - 1)])
axs[2].set_xlabel('{} (frames)'.format(key))
axs[0].set_ylabel('normalised APE size')
axs[2].set_ylabel('normalised APE size')
axs[1].set_ylabel('{} (frames)'.format(key))
axs[0].set_xlabel('trial type')
axs[1].set_xlabel('trial type')
plt.tight_layout()

fig, axs = plt.subplots(1, 1)
make_box_plot(q_data, dx='recording site', dy='r squared', fig_ax=axs)

colourmap=matplotlib.cm.inferno
colours = colourmap(np.linspace(0, 1, 8))
fig, axs = plt.subplots(1,1)
# for i in range(1, 8):
#     all_xs, all_ys = get_trial_type_data(trial_data, formatted_data, i, key='traces')
#     axs.plot(all_xs[0], color=colours[i])