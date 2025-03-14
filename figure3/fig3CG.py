from set_global_params import change_over_time_mice
import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import  *
from set_global_params import spreadsheet_path
import os

sh_path = os.path.join(spreadsheet_path, 'fig3')


# this makes the plot in the figure 3 C & G
font = {'size': 8}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(2, 1, figsize=[2, 4], constrained_layout=True)
tail_mice = change_over_time_mice['tail']
tail_ys = make_change_over_time_plot(tail_mice, ax[0], window_for_binning=50, colour='#002F3A', line='#002F3A')

nacc_mice = change_over_time_mice['Nacc']
nacc_ys = make_change_over_time_plot(nacc_mice, ax[1], window_for_binning=50, colour='#E95F32', line='#E95F32')

# save spreadsheet data
fn_nacc = 'fig3C_vs.csv'
np.savetxt(os.path.join(sh_path, fn_nacc), nacc_ys)
fn_tail = 'fig3G_ts.csv'
np.savetxt(os.path.join(sh_path, fn_tail), tail_ys)

makes_plots_pretty([ax[0], ax[1]])
plt.show()