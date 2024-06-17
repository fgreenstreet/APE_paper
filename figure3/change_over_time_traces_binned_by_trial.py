import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import *
font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']
bin_window = 200
fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=[2, 3.5])

make_example_traces_plot('SNL_photo17', axs[0], window_for_binning=bin_window, legend=False, align_to='movement')

make_example_traces_plot('SNL_photo35', axs[1], window_for_binning=bin_window, legend=False, align_to='cue')

makes_plots_pretty(axs)
plt.tight_layout()
plt.show()

