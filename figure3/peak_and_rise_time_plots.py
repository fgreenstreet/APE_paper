import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from set_global_params import processed_data_path
from utils.plotting_visuals import makes_plots_pretty
from utils.plotting import output_significance_stars_from_pval
from scipy.stats import ttest_ind
from utils.box_plot_utils import *

sites = ['Nacc', 'tail']
peak_times = {}
rise_times = {}
for site in sites:
    dir = processed_data_path + 'for_figure\\'
    file_name = 'peak_times_and_time_to_slope_ipsi_and_contra_{}.npz'.format(site)
    site_data = np.load(dir + file_name)
    peak_times[site] = site_data['peak_times']
    rise_times[site] = site_data['time_to_slope']

# Create a list of dictionaries with site information
peak_times_list = [{'site': site, 'peak time (s)': time} for site, times in peak_times.items() for time in times]
rise_times_list = [{'site': site, 'rise time (s)': time} for site, times in rise_times.items() for time in times]

# Convert the lists of dictionaries to DataFrames

peak_times_df = pd.DataFrame(peak_times_list).replace({'tail': 'TS', 'Nacc': 'VS'})
rise_times_df = pd.DataFrame(rise_times_list).replace({'tail': 'TS', 'Nacc': 'VS'})

# Set font and style parameters
font = {'size': 8.5, 'family': 'sans-serif', 'sans-serif': ['Arial']}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42


fig_directory = 'T:\\paper\\revisions\\cue movement reward comparisons VS TS\\'


# Plot peak time comparison
plot_and_save_comparison(peak_times_df, 'peak time (s)', fig_directory, 'peak_time_comparison_TS_VS_ipsi_and_contra.pdf')

# Plot rise time comparison
plot_and_save_comparison(rise_times_df, 'rise time (s)', fig_directory, 'rise_time_comparison_TS_VS_ipsi_and_contra.pdf')
plt.show()