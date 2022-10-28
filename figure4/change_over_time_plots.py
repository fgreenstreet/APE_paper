from set_global_params import change_over_time_mice
import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import  *

# this makes the plot in the figure
font = {'size': 8}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(2, 1, figsize=[2,4], constrained_layout=True)
tail_mice = change_over_time_mice['tail']
make_change_over_time_plot(tail_mice, ax[0], window_for_binning=50, colour='#002F3A', line='#002F3A')

nacc_mice = change_over_time_mice['Nacc']
make_change_over_time_plot(nacc_mice, ax[1], window_for_binning=50, colour='#E95F32', line='#E95F32')

makes_plots_pretty([ax[0], ax[1]])
plt.show()