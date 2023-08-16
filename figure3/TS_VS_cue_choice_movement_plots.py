import matplotlib.pylab as plt
from utils.zscored_plots_utils import get_data_for_recording_site, make_y_lims_same_heat_map, plot_all_heatmaps_same_scale, plot_average_trace_all_mice_cue_move_rew
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.colors
import numpy as np
import matplotlib
import cmocean as cmo

from utils.plotting_visuals import makes_plots_pretty
from set_global_params import figure_directory


font = {'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

fig = plt.figure(constrained_layout=True, figsize=[6, 2])
gs = fig.add_gridspec(nrows=1, ncols=3)

cue_ax = fig.add_subplot(gs[0, 0])
cue_ax.set_title('cue')
move_ax = fig.add_subplot(gs[0, 1])
move_ax.set_title('move')
reward_ax = fig.add_subplot(gs[0, 2])
reward_ax.set_title('reward')

all_axs = {'cue': [cue_ax],
            'contra': [move_ax],
            'rewarded': [reward_ax]}


colours = sns.color_palette("Set2")[:2]
#t_axs, t_data, t_wd, t_flip_sort_order, t_y_mins, t_y_maxs = get_data_for_recording_site('TS', all_axs)
#v_axs, v_data, v_wd, v_flip_sort_order, v_y_mins, v_y_maxs = get_data_for_recording_site('VS', all_axs)



plot_average_trace_all_mice_cue_move_rew(cue_ax, move_ax, reward_ax, error_bar_method='sem', cmap=sns.color_palette("Set2"), x_range=[-1.5, 1.5])
makes_plots_pretty([cue_ax, move_ax, reward_ax])
data_directory = 'T:\\paper\\revisions\\cue movement reward comparisons VS TS\\'
plt.savefig(data_directory + 'cue_movement_reward_aligned_traces_VS_TS_all_cues.pdf', transparent=True, bbox_inches='tight')

plt.tight_layout()
plt.show()