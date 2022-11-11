import os
import peakutils
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from utils.plotting import multi_conditions_plot
from utils.plotting_visuals import makes_plots_pretty
from scipy import stats
from utils.large_reward_omission_utils import plot_mean_trace_for_condition
from set_global_params import processed_data_path


site = 'Nacc'
mouse_name = 'SNL_photo31'

processed_data_dir = os.path.join(processed_data_path, 'large_rewards_omissions_data')
block_data_file = os.path.join(processed_data_dir, 'all_nacc_reward_change_data.csv')
all_reward_block_data = pd.read_pickle(block_data_file)

all_time_points = all_reward_block_data['time points'].reset_index(drop=True)[0]
start_plot = int(all_time_points.shape[0]/2 - 2*1000)
end_plot = int(all_time_points.shape[0]/2 + 2*1000)
time_points = all_time_points[start_plot: end_plot]
contra_data = all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name) & (all_reward_block_data['side'] == 'contra')]
ipsi_data =  all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name) & (all_reward_block_data['side'] == 'ipsi')]
contra_traces = np.vstack(contra_data['traces'].values)[:, start_plot: end_plot]
ipsi_traces= np.vstack(ipsi_data['traces'].values)[:, start_plot: end_plot]

contra_data = all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name) & (all_reward_block_data['side'] == 'contra')]
ipsi_data = all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name) & (all_reward_block_data['side'] == 'ipsi')]

font = {'size'   : 7}

matplotlib.rc('font', **font)


all_time_points = all_reward_block_data['time points'].reset_index(drop=True)[0]

contra_data = all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name) & (all_reward_block_data['side'] == 'contra')]
ipsi_data =  all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name) & (all_reward_block_data['side'] == 'ipsi')]


fig, ax = plt.subplots(2,1,figsize=[2.5, 4]) #, figsize=(10,16))
plot_mean_trace_for_condition(ax[0], contra_data, all_time_points,
                              'reward contra', error_bar_method='sem', save_location=processed_data_dir)
lg1 = ax[0].legend(loc='lower left', bbox_to_anchor=(0.6, 0.8),
            borderaxespad=0, frameon=False,prop={'size': 6 })

plot_mean_trace_for_condition(ax[1],ipsi_data, all_time_points,
                              'reward ipsi', error_bar_method = 'sem', save_location=processed_data_dir)
lg2 = ax[1].legend(loc='lower left', bbox_to_anchor=(0.6, 0.8),
            borderaxespad=0, frameon=False,prop={'size': 6 })
makes_plots_pretty(ax)
plt.tight_layout()

#plt.savefig(os.path.join(figure_dir, 'example_mouse{}.pdf'.format(mouse_name)))


font = {'size'   : 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

mouse_name = 'SNL_photo31'


timepoints = all_reward_block_data['time points'].reset_index(drop=True)[0]
all_trials = all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name)]

fig, ax = plt.subplots(1,1, figsize=[2.2, 2]) #, figsize=(10,16))
plot_mean_trace_for_condition(ax, all_trials, timepoints,
                              'reward', error_bar_method='sem', save_location=processed_data_dir)
lg1 = ax.legend(loc='lower left', bbox_to_anchor=(0.6, 0.8),
            borderaxespad=0, frameon=False,prop={'size': 6 })
ax.set_ylim([-1.5, 4.1])
makes_plots_pretty(ax)
plt.tight_layout()
figure_dir = 'W:\\paper'
#plt.savefig(os.path.join(figure_dir, 'example_mouse{}_both_sides.pdf'.format(mouse_name)))


# find mean traces and downsample
avg_traces = all_reward_block_data.groupby(['mouse', 'reward'])['traces'].apply(np.mean)
decimated = [decimate(trace[int(len(trace)/2):], 10) for trace in avg_traces]
avg_traces = avg_traces.reset_index()
avg_traces['decimated'] = pd.Series([_ for _ in decimated])



first_peak_ids = [peakutils.indexes(i)[0] for i in avg_traces['decimated']]
avg_traces['peakidx'] = first_peak_ids
peaks = [np.mean(trace[:600]) for idx, trace in zip(first_peak_ids, avg_traces['decimated'])]
avg_traces['peak'] = peaks
avg_traces.set_index(['mouse', 'reward'])

normal_peak = avg_traces[avg_traces['reward']=='normal']['peak']
large_reward_peak = avg_traces[avg_traces['reward']=='large reward']['peak']
omission_peak = avg_traces[avg_traces['reward']=='omission']['peak']
stat1, pval1 = stats.ttest_rel(normal_peak, large_reward_peak)
stat2, pval2 = stats.ttest_rel(normal_peak, omission_peak)

# We run a repeated measures anova to check for a main effect of reward.
# Subsequently, we want to do pairwise testing between the three reward conditions. Need to correct for multiple comparisons
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

reject, corrected_pvals, corrected_alpha_sidak, corrected_bonf = multipletests([pval1, pval2], method='bonferroni')

print(corrected_pvals)

df1 = avg_traces
df_for_plot = df1.pivot(index='reward', columns='mouse', values='peak').sort_values('reward', ascending=False)
font = {'size'   : 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

fig, ax = plt.subplots(figsize=[2,2])
multi_conditions_plot(ax, df_for_plot, mean_line_color='#7FB5B5', mean_linewidth=3, show_err_bar=False)
plt.xticks([0, 1, 2], ['omission', 'normal\nreward', '3 x normal\nreward'], fontsize=7)
plt.ylabel('Z-scored fluorescence', fontsize=7)
ax.set_xlabel(' ')
#ax.text(1.2, 3, 'p-value = {0:.6f}'.format(corrected_pvals[1]))
#ax.text(0.1, 3, 'p-value = {0:.6f}'.format(corrected_pvals[0]))

# show significance stars
# for first comparison
y = df_for_plot.T['large reward'].max() + .2
h = .1
plt.plot([0, 0, 1, 1], [y, y+h, y+h, y],c='k',lw=1)
#ax.text(.5, y+h, 'n.s.', ha='center', fontsize=8)
#ax.text(.5, y+h, 'n.s.', ha='center', fontsize=8)
ax.text(.5, y+h, '****', ha='center', fontsize=8)
# for second comparison
l = .2
plt.plot([1, 1, 2, 2], [y+l, y+h+l, y+h+l, y+l],c='k', linewidth=1)
ax.text(1.5, y+h+l, '****', ha='center', fontsize=8)
ax.set_ylim([-1, 3.4])
plt.tight_layout()
filepath=os.path.join('W:\\paper', 'group_data_omissions_large_rewards_{}.pdf'.format(site))
fig.savefig(filepath)
