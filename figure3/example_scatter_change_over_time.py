import matplotlib
from utils.plotting_visuals import makes_plots_pretty
import matplotlib.pyplot as plt
from utils.change_over_time_plot_utils import example_scatter_change_over_time

font = {'size': 7}
matplotlib.rc('font', **font)
fig, axs = plt.subplots(2, 1, figsize=[2, 4])
example_scatter_change_over_time('SNL_photo17', axs[0], window_for_binning=40, colour='#002F3A')
example_scatter_change_over_time('SNL_photo31', axs[1], window_for_binning=40, colour='#E95F32')
makes_plots_pretty(axs)
plt.show()