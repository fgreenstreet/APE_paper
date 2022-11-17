import peakutils
from matplotlib import colors, cm
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
from scipy import stats
from utils.post_processing_utils import open_one_experiment

from utils.value_change_utils import CustomAlignedDataRewardBlocks, get_all_experimental_records, plot_mean_trace_for_condition, get_block_change_info, add_traces_and_peaks, one_session_get_block_changes

site= 'Nacc' #'tail'

nacc_file = 'value_switch_nacc_mice.csv'
tail_file = 'value_switch_all_tail_mice_test_new_mice_added.csv'
root_directory = 'W:\\photometry_2AC'
processed_data_dir = os.path.join(root_directory, 'value_change_data')
block_data_file = os.path.join(processed_data_dir, nacc_file)
all_reward_block_data = pd.read_pickle(block_data_file)

all_time_points = all_reward_block_data['time points'].iloc[0]
font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']


#example mouse
mouse_name = 'SNL_photo70'
data = all_reward_block_data[all_reward_block_data.mouse == mouse_name]


fig, ax = plt.subplots(1,1, figsize=(2.5,2)) #, figsize=(10,16))
possible_values = np.array([-4, 0, 4])
plot_mean_trace_for_condition(ax, data, all_time_points,
                              'relative reward amount', error_bar_method = 'sem', save_location=processed_data_dir)
lg2 = ax.legend(title='Relative \n value (ul)',loc='lower left', bbox_to_anchor=(0.7, 0.7),
            borderaxespad=0, frameon=False,prop={'size': 6 })
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(loc='lower left', bbox_to_anchor=(0.7, 0.8),
            borderaxespad=0, frameon=False,prop={'size': 6 })
plt.tight_layout()
filepath=os.path.join('W:\\paper', 'example_value_change_{}_{}.pdf'.format(mouse_name,site) )
#fig.savefig(filepath)


#group data
sorted_data = all_reward_block_data.sort_values(['mouse', 'session', 'trial number']).reset_index(drop=True)
sorted_data['block switches'] = sorted_data['block number'] - sorted_data['block number'].shift()
sorted_data['new sessions'] = sorted_data['session'].ne(sorted_data['session'].shift().bfill()).astype(int)
sorted_data.iloc[0, sorted_data.columns.get_loc('new sessions')] = 1
sorted_data.loc[sorted_data['new sessions'] == 1, 'block switches'] = 1
block_switch_inds = sorted_data.loc[sorted_data['block switches'] != 0].reset_index(drop=True)

num_trials_to_look_at = 50
min_num_trials = 70
traces = []
peaks = []
trial_nums = []
block_inds = []
rel_reward_amounts = []
reward_amounts = []
mouse_ids = []
session_ids = []
num_blocks = block_switch_inds.index.values.shape[0]
for block_num, block in block_switch_inds.iterrows():
    first_trial = block['trial number']
    mouse = block['mouse']
    session = block['session']
    block_id = block['block number']
    block_switch_trial_num = block['trial number']
    all_session_trials = sorted_data[(sorted_data['mouse'] == mouse) & (sorted_data['session'] == session)]
    all_block_trials = all_session_trials[all_session_trials['block number'] == block_id]
    if block_id == 0 or all_block_trials.shape[0]>=min_num_trials:
        last_trials_of_block = all_block_trials[-num_trials_to_look_at:]
        mean_trace = last_trials_of_block['traces'].apply(np.mean)
        avg_trace = last_trials_of_block.groupby(['mouse', 'contra reward amount'])['traces'].apply(np.mean).values[0]
        decimated = decimate(avg_trace[int(len(avg_trace)/2):], 10)
        peak_idx = peakutils.indexes(decimated)[0]
        peak = decimated[peak_idx]
        traces.append(decimated)
        peaks.append(peak)
        trial_nums.append(last_trials_of_block['trial number'].values)
        rel_reward_amounts.append(last_trials_of_block['relative reward amount'].values[0])
        reward_amounts.append(last_trials_of_block['contra reward amount'].values[0])
        block_inds.append(last_trials_of_block.index.values[0])
        mouse_ids.append(mouse)
        session_ids.append(session)

avg_block_data = {}
avg_block_data['block id'] = block_inds
avg_block_data['peaks'] = peaks
avg_block_data['relative reward amount'] = rel_reward_amounts
avg_block_data['contra reward amount'] = reward_amounts
avg_block_data['mouse'] = mouse_ids
avg_block_data['session'] = session_ids
avg_block_dataf = pd.DataFrame(avg_block_data)

avg_block_dataf['avg traces'] = pd.Series(traces, index=avg_block_dataf.index)
df_for_plot = avg_block_dataf.groupby(['mouse', 'session', 'relative reward amount', 'contra reward amount'])['peaks'].apply(np.mean)
df_for_plot = df_for_plot.reset_index()

mice = []
reward_changes = []
cue_changes = []
session_ids = []

first_blocks = df_for_plot[df_for_plot['relative reward amount'] == 0]
for first_block_ind, first_block in first_blocks.iterrows():
    mouse = first_block['mouse']
    session = first_block['session']
    session_blocks = df_for_plot[np.logical_and(df_for_plot['mouse'] == mouse, df_for_plot['session'] == session)]
    other_block = session_blocks.loc[session_blocks['relative reward amount'] != 0]
    if other_block.shape[0] > 0:
        mice.append(mouse)
        reward_changes.append(other_block['relative reward amount'].values[0])
        cue_change = (other_block['peaks'] - first_block['peaks']).values[0]
        cue_changes.append(cue_change)
        session_ids.append(session)
diff_block_data = {}
diff_block_data['mouse'] = mice
diff_block_data['reward size change'] = reward_changes
diff_block_data['cue size change'] = cue_changes
diff_block_data['session'] = session_ids
diff_block_dataf = pd.DataFrame(diff_block_data)

df_for_plot1 = diff_block_dataf.groupby(['mouse', 'reward size change'])['cue size change'].apply(np.mean)
df_for_plot1 = df_for_plot1.reset_index()

df1 = df_for_plot1.pivot(index='reward size change', columns='mouse', values='cue size change').sort_values('reward size change', ascending=False)

small_data = df_for_plot1[(df_for_plot1['reward size change'] == -4)]['cue size change'].values
big_data = df_for_plot1[(df_for_plot1['reward size change'] == 4)]['cue size change'].values
stat, pval = stats.ttest_rel(small_data, big_data)
print(pval)

fg = sns.FacetGrid(data=df_for_plot1, hue='mouse', aspect=0.8, palette='Pastel1_d', height=3)
fg.map(plt.scatter, 'reward size change', 'cue size change', marker='o', s=30, facecolor='none')
ax = fg.axes[0][0]
ax.axhline(0, color='k', lw=0.5)
ax.set_xticks([-4, 4])
ax.set_xlabel(r'relative value ($\mu$l)')
ax.set_xlim([-6, 6])
plt.ylabel(r'$\Delta$ z-score')

y = df1.to_numpy().max() + .2
h = .05
plt.plot([-4, -4, 4, 4], [y, y+h, y+h, y],c='k',lw=1)
ax.text(0, y+h, '***', ha='center', fontsize=10)
plt.tight_layout()