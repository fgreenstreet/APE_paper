import os
import pandas as pd
from set_global_params import processed_data_path
from utils.box_plot_utils import *

filename = os.path.join(processed_data_path, 'num_pokes_in_punishment.pkl')

pre_silence_poke_count = pd.read_pickle(filename)

font = {'size': 8.5, 'family': 'sans-serif', 'sans-serif': ['Arial']}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42


fig_directory = 'T:\\paper\\revisions\\silent pokes pre silence\\'


# Plot peak time comparison
plot_and_save_comparison(pre_silence_poke_count, 'count', fig_directory, 'pokes_pre_silence.pdf', dx='stimulus', sig_test=False)
