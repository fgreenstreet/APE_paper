import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pickle
import matplotlib
from utils.post_processing_utils import remove_manipulation_days, remove_bad_recordings, remove_exps_after_manipulations_not_including_psychometric, get_all_experimental_records, remove_experiments
from utils.beginning_and_end_quantification_utils import get_first_and_10000th_peaks, get_mean_contra_peak, make_beginning_and_end_comparison_plot
from utils.plotting import two_conditions_plot

font = {'size': 6}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(2, 1, figsize=[2, 4])
make_beginning_and_end_comparison_plot(axs[0], site='tail', colour='#002F3A')
make_beginning_and_end_comparison_plot(axs[1], site='Nacc', colour='#E95F32')

plt.show()