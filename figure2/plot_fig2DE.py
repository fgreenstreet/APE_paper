import os
from set_global_params import all_data_path, reproduce_figures_path, processed_data_path
import matplotlib.pylab as plt
from utils.data_loading_utils import load_or_get_and_save_example_heatmap_aligned_to_keys
from utils.zscored_plots_utils import plot_all_heatmaps_same_scale, plot_average_trace_all_mice
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib
import cmocean
from utils.plotting_visuals import makes_plots_pretty
from scipy.signal import decimate
import pandas as pd
from functools import partial

font = {'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'


pickle_folder = os.path.join(all_data_path, 'reproducing_figures', 'fig2')
spreadsheet_folder = os.path.join(all_data_path, 'spreadsheets_for_nature', 'fig2')
if not os.path.exists(pickle_folder):
    os.makedirs(pickle_folder)
if not os.path.exists(spreadsheet_folder):
    os.makedirs(spreadsheet_folder)


fig = plt.figure(constrained_layout=True, figsize=[8, 4])
gs = fig.add_gridspec(nrows=2, ncols=4)

#cmap_map = matplotlib.cm.get_cmap('cmo.solar') #('cet_fire')#cc.cm['linear_wyor_100_45_c55']) #cc.cm['linear_tritanopic_krw_5_95_c46']) cmo.solar
cmap_map = cmocean.cm.solar
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

ts_pickle_path = os.path.join(pickle_folder, 'ts_example_heatmap_data.pkl')
vs_pickle_path = os.path.join(pickle_folder, 'vs_example_heatmap_data.pkl')
time_window_size = 2  # only save [-2:+2] seconds of data around each cue / movement onset

tail_hm_data = load_or_get_and_save_example_heatmap_aligned_to_keys(ts_pickle_path, 'TS', list(ts_heat_map_axs.keys()),
                                                                    time_window_size=time_window_size)
vs_hm_data = load_or_get_and_save_example_heatmap_aligned_to_keys(vs_pickle_path, 'VS', list(vs_heat_map_axs.keys()),
                                                                  time_window_size=time_window_size)

t_data, t_wd, t_flip_sort_order, t_y_mins, t_y_maxs = tail_hm_data
v_data, v_wd, v_flip_sort_order, v_y_mins, v_y_maxs = vs_hm_data

t_axs = [a[0] for a in ts_heat_map_axs.values()]
v_axs = [a[0] for a in vs_heat_map_axs.values()]
heat_map_t = plot_all_heatmaps_same_scale(t_axs, t_data, t_wd, t_flip_sort_order, [np.min(t_y_mins), np.max(t_y_maxs)],
                                          cmap=blue_cmap)
heat_map_v = plot_all_heatmaps_same_scale(v_axs, v_data, v_wd, v_flip_sort_order, [np.min(v_y_mins), np.max(v_y_maxs)],
                                          cmap=red_cmap)

heatmap_data = [hm.get_array() for hm in heat_map_t + heat_map_v]
# downsample because of nature's storage limit
heatmap_data = [decimate(hm, 50, axis=-1) for hm in heatmap_data]

areas = ['VS'] * 2 + ['TS'] * 2
sides = ['ipsi', 'contra'] * 2
for i, hm in enumerate(heatmap_data):
    fn = f'fig2D_heatmap_data_{areas[i]}_{sides[i]}.csv'
    fp = os.path.join(spreadsheet_folder, fn)
    hm_df = pd.DataFrame(hm.T)
    hm_df.columns = [f'Trial_{i}' for i in range(hm_df.shape[1])]
    hm_df.to_csv(fp)

# get average traces across mice
if os.path.exists(os.path.join(reproduce_figures_path, 'fig2')):
    print('loading cached group data')
    dir = os.path.join(reproduce_figures_path, 'fig2')
else:
    dir = os.path.join(processed_data_path, 'for_figure')

vs_outcome_data, vs_move_data = plot_average_trace_all_mice(vs_average_move_ax,  vs_average_outcome_ax, dir, 'Nacc', cmap=['#E95F32', '#F9C0AF'])
ts_outcome_data, ts_move_data = plot_average_trace_all_mice(ts_average_move_ax,  ts_average_outcome_ax, dir, 'tail', cmap=['#002F3A', '#76A8DA'])

makes_plots_pretty([vs_average_move_ax, ts_average_move_ax, vs_average_outcome_ax, ts_average_outcome_ax])
plt.tight_layout()

####################### write to csvs for nature
ds = partial(decimate, q=10)

vs_outcome_df = pd.DataFrame()
vs_outcome_df['time'] = ds(vs_outcome_data['time'])
for m in range(vs_outcome_data['data'][0].shape[0]):
    vs_outcome_df[f'reward_m{m}'] = ds(vs_outcome_data['data'][0][m])
    vs_outcome_df[f'no_reward_m{m}'] = ds(vs_outcome_data['data'][1][m])
vs_outcome_df.to_csv(os.path.join(spreadsheet_folder, 'fig2E_vs_outcome_avg_traces.csv'))

vs_move_df = pd.DataFrame()
vs_move_df['time'] = ds(vs_move_data['time'])
for m in range(vs_move_data['data'][0].shape[0]):
    vs_move_df[f'contra_m{m}'] = ds(vs_move_data['data'][0][m])
    vs_move_df[f'ipsi_m{m}'] = ds(vs_move_data['data'][1][m])
vs_move_df.to_csv(os.path.join(spreadsheet_folder, 'fig2E_vs_move_avg_traces.csv'))

ts_outcome_df = pd.DataFrame()
ts_outcome_df['time'] = ds(ts_outcome_data['time'])
for m in range(vs_outcome_data['data'][0].shape[0]):
    ts_outcome_df[f'reward_m{m}'] = ds(ts_outcome_data['data'][0][m])
    ts_outcome_df[f'noreward_m{m}'] = ds(ts_outcome_data['data'][1][m])
ts_outcome_df.to_csv(os.path.join(spreadsheet_folder, 'fig2E_ts_outcome_avg_traces.csv'))

ts_move_df = pd.DataFrame()
ts_move_df['time'] = ds(ts_move_data['time'])
for m in range(ts_move_data['data'][0].shape[0]):
    ts_move_df[f'contra_m{m}'] = ds(ts_move_data['data'][0][m])
    ts_move_df[f'ipsi_m{m}'] = ds(ts_move_data['data'][1][m])
ts_move_df.to_csv(os.path.join(spreadsheet_folder, 'fig2E_ts_move_avg_traces.csv'))


#plt.savefig(os.path.join(figure_directory, 'average_traces_figure_with_ttests.pdf'), transparent=True, bbox_inches='tight')
plt.show()

