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


cmap_map = matplotlib.cm.get_cmap('cmo.solar') #('cet_fire')#cc.cm['linear_wyor_100_45_c55']) #cc.cm['linear_tritanopic_krw_5_95_c46']) cmo.solar
cmap_values = cmap_map(np.linspace(0, 1, 1200))

new_cmap_values = np.copy(cmap_values)
new_cmap_values[:, 2] = np.copy(cmap_values[:, 0])
new_cmap_values[:, 0] = np.copy(cmap_values[:, 2])

blue_cmap = ListedColormap(new_cmap_values)
red_cmap = ListedColormap(cmap_values)

# heatmaps
fig, axs = plt.subplots(1, 2, figsize=[4, 2])
ts_ax = axs[0]
ts_ax.set_title('cue TS')
vs_ax = axs[1]
vs_ax.set_title('cue VS')

ts_heat_map_axs = {'cue': [ts_ax]}
vs_heat_map_axs = {'cue': [vs_ax]}

colours = sns.color_palette("Set2")[:2]
t_axs, t_data, t_wd, t_flip_sort_order, t_y_mins, t_y_maxs = get_data_for_recording_site('TS', ts_heat_map_axs)
v_axs, v_data, v_wd, v_flip_sort_order, v_y_mins, v_y_maxs = get_data_for_recording_site('VS', vs_heat_map_axs)

heat_map_t = plot_all_heatmaps_same_scale(t_axs, t_data, t_wd, t_flip_sort_order, (np.min(t_y_mins), np.max(t_y_maxs)), cmap=blue_cmap)
heat_map_v = plot_all_heatmaps_same_scale(v_axs, v_data, v_wd, v_flip_sort_order, (np.min(v_y_mins), np.max(v_y_maxs)), cmap=red_cmap)
plt.tight_layout()
plt.show()
data_directory = 'T:\\paper\\revisions\\cue movement reward comparisons VS TS\\'
plt.savefig(data_directory + 'cue_aligned_heatmap_VS_TS.pdf', transparent=True, bbox_inches='tight')
