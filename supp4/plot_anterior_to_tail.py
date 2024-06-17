import matplotlib.pylab as plt
from utils.zscored_plots_utils import get_data_for_recording_site, make_y_lims_same_heat_map, plot_all_heatmaps_same_scale, plot_average_trace_all_mice
import matplotlib


from utils.plotting_visuals import makes_plots_pretty

font = {'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

fig = plt.figure(constrained_layout=True, figsize=[4, 2])
gs = fig.add_gridspec(nrows=1, ncols=2)


ant_tail_average_move_ax = fig.add_subplot(gs[0, 0])
ant_tail_average_move_ax.set_title('Choice')
ant_tail_average_outcome_ax = fig.add_subplot(gs[0, 1])
ant_tail_average_outcome_ax.set_title('Outcome')


plot_average_trace_all_mice(ant_tail_average_move_ax,  ant_tail_average_outcome_ax, 'tail_ant', cmap=['#E95F32', '#F9C0AF'])

makes_plots_pretty([ant_tail_average_move_ax,  ant_tail_average_outcome_ax])
plt.tight_layout()

plt.show()

