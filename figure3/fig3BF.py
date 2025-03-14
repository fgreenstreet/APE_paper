import matplotlib
from utils.plotting_visuals import makes_plots_pretty
import matplotlib.pyplot as plt
from utils.change_over_time_plot_utils import example_scatter_change_over_time
from save_to_excel import save_figure_data_to_excel
from set_global_params import spreadsheet_path
import os

sh_path = os.path.join(spreadsheet_path, 'fig3')
sh_fn = 'fig3BF_data.xlsx'

font = {'size': 7}
matplotlib.rc('font', **font)
fig, axs = plt.subplots(2, 1, figsize=[2, 4])
example_scatter_change_over_time('SNL_photo17', axs[0], window_for_binning=40, colour='#002F3A')
example_scatter_change_over_time('SNL_photo31', axs[1], window_for_binning=40, colour='#E95F32')
makes_plots_pretty(axs)

save_figure_data_to_excel(fig,os.path.join(sh_path, sh_fn))
plt.show()