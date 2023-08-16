import sys

import matplotlib.pylab as plt
from utils.zscored_plots_utils import get_data_for_recording_site, make_y_lims_same_heat_map, plot_all_heatmaps_same_scale, plot_average_trace_all_mice
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

fig = plt.figure(constrained_layout=True, figsize=[8, 4])
gs = fig.add_gridspec(nrows=2, ncols=4)

cmap_map = matplotlib.cm.get_cmap('cmo.solar') #('cet_fire')#cc.cm['linear_wyor_100_45_c55']) #cc.cm['linear_tritanopic_krw_5_95_c46']) cmo.solar
cmap_values = cmap_map(np.linspace(0, 1, 1200))

new_cmap_values = np.copy(cmap_values)
new_cmap_values[:, 2] = np.copy(cmap_values[:, 0])
new_cmap_values[:, 0] = np.copy(cmap_values[:, 2])

blue_cmap = ListedColormap(new_cmap_values)
red_cmap = ListedColormap(cmap_values)

ts_heatmap_contra_ax = fig.add_subplot(gs[0, 0])
ts_heatmap_contra_ax.set_title('TS contra')
ts_heatmap_ipsi_ax = fig.add_subplot(gs[0, 1])
ts_heatmap_ipsi_ax.set_title('TS ipsi')

vs_heatmap_contra_ax = fig.add_subplot(gs[1, 0])
vs_heatmap_contra_ax.set_title('VS contra')
vs_heatmap_ipsi_ax = fig.add_subplot(gs[1, 1])
vs_heatmap_ipsi_ax.set_title('VS ipsi')

vs_average_move_ax = fig.add_subplot(gs[1, 2])
vs_average_move_ax.set_title('Choice')
vs_average_outcome_ax = fig.add_subplot(gs[1, 3])
vs_average_outcome_ax.set_title('Outcome')


ts_average_move_ax = fig.add_subplot(gs[0, 2],sharey=vs_average_move_ax)
ts_average_move_ax.set_title('Choice')
ts_average_outcome_ax = fig.add_subplot(gs[0, 3], sharey=vs_average_outcome_ax)
ts_average_outcome_ax.set_title('Outcome')

ts_heat_map_axs = {'contra': [ts_heatmap_contra_ax],
                   'ipsi': [ts_heatmap_ipsi_ax]}
vs_heat_map_axs = {'contra': [vs_heatmap_contra_ax],
                   'ipsi': [vs_heatmap_ipsi_ax]}

colours = sns.color_palette("Set2")[:2]
t_axs, t_data, t_wd, t_flip_sort_order, t_y_mins, t_y_maxs = get_data_for_recording_site('TS', ts_heat_map_axs)
v_axs, v_data, v_wd, v_flip_sort_order, v_y_mins, v_y_maxs = get_data_for_recording_site('VS', vs_heat_map_axs)

heat_map_t = plot_all_heatmaps_same_scale(t_axs, t_data, t_wd, t_flip_sort_order, (np.min(t_y_mins), np.max(t_y_maxs)), cmap=blue_cmap)
heat_map_v = plot_all_heatmaps_same_scale(v_axs, v_data, v_wd, v_flip_sort_order, (np.min(v_y_mins), np.max(v_y_maxs)), cmap=red_cmap)

plot_average_trace_all_mice(vs_average_move_ax,  vs_average_outcome_ax, 'Nacc', cmap=['#E95F32', '#F9C0AF'])
plot_average_trace_all_mice(ts_average_move_ax,  ts_average_outcome_ax, 'tail', cmap=['#002F3A', '#76A8DA'])

makes_plots_pretty([vs_average_move_ax, ts_average_move_ax, vs_average_outcome_ax, ts_average_outcome_ax])
plt.tight_layout()

#plt.savefig(figure_directory + 'Fig3_paper_final_version.pdf', transparent=True, bbox_inches='tight')
plt.show()