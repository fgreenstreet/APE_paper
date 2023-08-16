from utils.zscored_plots_utils import get_all_mouse_data_for_site
from utils.peak_and_rise_utils import get_peak_times, get_DA_peak_times_and_slopes_from_cue
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from utils.post_processing_utils import remove_exps_after_manipulations, get_first_x_sessions
from set_global_params import experiment_record_path, processed_data_path, mice_average_traces
from utils.average_trace_processing_utils import get_all_mice_average_data_only_contra_cues

#
# sites = ['tail', 'Nacc']
# peak_times = {}
# for site in sites:
#     all_data = get_all_mouse_data_for_site(site, file_ext='_new_mice_added_with_cues.npz')
#     time_stamps = all_data['time_stamps']
#     data = dict(all_data)
#     cue_traces = data['cue']
#     peak_times[site] = (get_peak_times(cue_traces, time_stamps))
# print('a')
sites = ['tail', 'Nacc']
peak_times = {}
trace_slopes = {}
for site in sites:
    mouse_ids = mice_average_traces[site]
    experiment_record = pd.read_csv(experiment_record_path, dtype='str')
    experiments_to_process = get_first_x_sessions(experiment_record, mouse_ids, site).reset_index(drop=True)
    peak_times[site], trace_slopes[site] = get_DA_peak_times_and_slopes_from_cue(experiments_to_process)
    # save out these stats for plotting etc
    dir = processed_data_path + 'for_figure\\'
    file_name = 'peak_times_and_time_to_slope_{}.npz'.format(site)
    np.savez(dir + file_name, peak_times=peak_times[site], time_to_slope=trace_slopes[site])
print(ttest_ind(trace_slopes['Nacc'], trace_slopes['tail']))
print(ttest_ind(peak_times['Nacc'], peak_times['tail']))

plt.figure()
plt.hist(trace_slopes['Nacc'], alpha=0.5)
plt.hist(trace_slopes['tail'], alpha=0.5)

plt.figure()
plt.hist(peak_times['Nacc'], alpha=0.5)
plt.hist(peak_times['tail'], alpha=0.5)
