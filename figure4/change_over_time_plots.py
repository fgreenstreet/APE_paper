from scipy.interpolate import interp1d
from utils.plotting import calculate_error_bars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
from matplotlib.lines import Line2D
import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import  *
# this makes the plot in the figure
font = {'size': 8}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(2, 1, figsize=[2,4], constrained_layout=True)
tail_mice = ['SNL_photo16', 'SNL_photo17','SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']
make_change_over_time_plot(tail_mice, ax[0], window_for_binning=50, colour='#002F3A', line='#002F3A')

nacc_mice = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
make_change_over_time_plot(nacc_mice, ax[1], window_for_binning=50, colour='#E95F32', line='#E95F32')


makes_plots_pretty([ax[0], ax[1]])
plt.show()