import os
from set_global_params import all_data_path
import matplotlib.pylab as plt
from utils.zscored_plots_utils import get_example_data_for_recording_site, make_y_lims_same_heat_map, plot_all_heatmaps_same_scale, plot_average_trace_all_mice
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib
import cmocean
import pickle
from utils.plotting_visuals import makes_plots_pretty
from set_global_params import figure_directory
from scipy.signal import decimate
import pandas as pd
from functools import partial

font = {'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

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
time_window_size = 2

# Try to load TS data from pickle
if os.path.exists(ts_pickle_path):
    print(f"Loading cached TS data from {ts_pickle_path}")
    with open(ts_pickle_path, 'rb') as f:
        ts_cache = pickle.load(f)
        t_data = ts_cache['data']
        t_wd = ts_cache['wd']
        t_flip_sort_order = ts_cache['flip_sort_order']
        t_y_mins = ts_cache['y_mins']
        t_y_maxs = ts_cache['y_maxs']
else:
    # Get data and save to pickle
    print("Loading TS data from raw and saving to cache...")
    t_data, t_wd, t_flip_sort_order, t_y_mins, t_y_maxs = get_example_data_for_recording_site('TS', list(ts_heat_map_axs.keys()))

    for obj in t_data:
        bool_idx = (obj.time_points < time_window_size) & (obj.time_points >= -time_window_size)
        obj.sorted_traces = obj.sorted_traces[:, bool_idx]
        obj.time_points = obj.time_points[bool_idx]

    # Save to pickle
    ts_cache = {
        'data': t_data,
        'wd': t_wd,
        'flip_sort_order': t_flip_sort_order,
        'y_mins': t_y_mins,
        'y_maxs': t_y_maxs
    }
    with open(ts_pickle_path, 'wb') as f:
        pickle.dump(ts_cache, f)

# Try to load VS data from pickle
if os.path.exists(vs_pickle_path):
    print(f"Loading cached VS data from {vs_pickle_path}")
    with open(vs_pickle_path, 'rb') as f:
        vs_cache = pickle.load(f)
        v_data = vs_cache['data']
        v_wd = vs_cache['wd']
        v_flip_sort_order = vs_cache['flip_sort_order']
        v_y_mins = vs_cache['y_mins']
        v_y_maxs = vs_cache['y_maxs']
else:
    # Get data and save to pickle
    print("Loading VS data from raw and saving to cache...")
    v_data, v_wd, v_flip_sort_order, v_y_mins, v_y_maxs = get_example_data_for_recording_site('VS', list(vs_heat_map_axs.keys()))

    for obj in v_data:
        bool_idx = (obj.time_points < time_window_size) & (obj.time_points >= -time_window_size)
        obj.sorted_traces = obj.sorted_traces[:, bool_idx]
        obj.time_points = obj.time_points[bool_idx]

    # Save to pickle
    vs_cache = {
        'data': v_data,
        'wd': v_wd,
        'flip_sort_order': v_flip_sort_order,
        'y_mins': v_y_mins,
        'y_maxs': v_y_maxs
    }
    with open(vs_pickle_path, 'wb') as f:
        pickle.dump(vs_cache, f)

t_axs = [a[0] for a in ts_heat_map_axs.values()]
v_axs = [a[0] for a in vs_heat_map_axs.values()]
heat_map_t = plot_all_heatmaps_same_scale(t_axs, t_data, t_wd, t_flip_sort_order, (np.min(t_y_mins), np.max(t_y_maxs)), cmap=blue_cmap)
heat_map_v = plot_all_heatmaps_same_scale(v_axs, v_data, v_wd, v_flip_sort_order, (np.min(v_y_mins), np.max(v_y_maxs)), cmap=red_cmap)

heatmap_data = [hm.get_array() for hm in heat_map_t + heat_map_v]
# downsample because of nature's storage limit
heatmap_data = [decimate(hm, 50, axis=-1) for hm in heatmap_data]

areas = ['VS'] * 2 + ['TS'] * 2
sides = ['ipsi', 'contra'] * 2
for i, hm in enumerate(heatmap_data):
    fn = f'heatmap_data_{areas[i]}_{sides[i]}.csv'
    fp = os.path.join(spreadsheet_folder, fn)
    np.savetxt(fp, hm)

vs_outcome_data, vs_move_data = plot_average_trace_all_mice(vs_average_move_ax,  vs_average_outcome_ax, 'Nacc', cmap=['#E95F32', '#F9C0AF'])
ts_outcome_data, ts_move_data = plot_average_trace_all_mice(ts_average_move_ax,  ts_average_outcome_ax, 'tail', cmap=['#002F3A', '#76A8DA'])

makes_plots_pretty([vs_average_move_ax, ts_average_move_ax, vs_average_outcome_ax, ts_average_outcome_ax])
plt.tight_layout()

####################### write to csvs for nature
ds = partial(decimate, q=10)

vs_outcome_df = pd.DataFrame()
vs_outcome_df['time'] = ds(vs_outcome_data['time'])
for m in range(vs_outcome_data['data'][0].shape[0]):
    vs_outcome_df[f'reward_m{m}'] = ds(vs_outcome_data['data'][0][m])
    vs_outcome_df[f'no_reward_m{m}'] = ds(vs_outcome_data['data'][1][m])
vs_outcome_df.to_csv(os.path.join(spreadsheet_folder, 'vs_outcome_avg_traces.csv'))

vs_move_df = pd.DataFrame()
vs_move_df['time'] = ds(vs_move_data['time'])
for m in range(vs_move_data['data'][0].shape[0]):
    vs_move_df[f'contra_m{m}'] = ds(vs_move_data['data'][0][m])
    vs_move_df[f'ipsi_m{m}'] = ds(vs_move_data['data'][1][m])
vs_move_df.to_csv(os.path.join(spreadsheet_folder, 'vs_move_avg_traces.csv'))

ts_outcome_df = pd.DataFrame()
ts_outcome_df['time'] = ds(ts_outcome_data['time'])
for m in range(vs_outcome_data['data'][0].shape[0]):
    ts_outcome_df[f'reward_m{m}'] = ds(ts_outcome_data['data'][0][m])
    ts_outcome_df[f'noreward_m{m}'] = ds(ts_outcome_data['data'][1][m])
ts_outcome_df.to_csv(os.path.join(spreadsheet_folder, 'ts_outcome_avg_traces.csv'))

ts_move_df = pd.DataFrame()
ts_move_df['time'] = ds(ts_move_data['time'])
for m in range(ts_move_data['data'][0].shape[0]):
    ts_move_df[f'contra_m{m}'] = ds(ts_move_data['data'][0][m])
    ts_move_df[f'ipsi_m{m}'] = ds(ts_move_data['data'][1][m])
ts_move_df.to_csv(os.path.join(spreadsheet_folder, 'ts_move_avg_traces.csv'))




#plt.savefig(os.path.join(figure_directory, 'average_traces_figure_with_ttests.pdf'), transparent=True, bbox_inches='tight')
plt.show()

