import os
from utils.plotting_visuals import set_plotting_defaults
import matplotlib.pyplot as plt
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.plotting import calculate_error_bars
from utils.return_to_centre_regression_utils import get_first_x_sessions_reg_rtc
import pandas as pd
from utils.tracking_analysis.camera_trigger_preprocessing_utils import *
from scipy.signal import decimate


mouse_ids = ['SNL_photo57', 'SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
num_sessions = 3
site = 'tail'

experiment_record = pd.read_csv('T:\\photometry_2AC\\experimental_record.csv')
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
all_experiments_to_process = all_experiments_to_process[all_experiments_to_process['include return to centre'] != 'no'].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process, x=num_sessions).reset_index(
    drop=True)
all_mice_contra_traces = np.zeros([len(mouse_ids), 100000])
all_mice_ipsi_traces = np.zeros([len(mouse_ids), 100000])

i = 0
time_points = np.linspace(-5, 5, 100000)
for mouse_num, mouse in enumerate(mouse_ids):
    experiments = experiments_to_process[experiments_to_process['mouse_id'] == mouse].reset_index(drop=True)
    contra_movement_traces = np.zeros([num_sessions, 100000])
    ipsi_movement_traces = np.zeros([num_sessions, 100000])
    for index, experiment in experiments.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        save_dir = 'T:\\photometry_2AC\\processed_data\\return_to_centre\\{}'.format(mouse)
        save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_300frame_window.npz'.format(mouse, date)
        traces = np.load(os.path.join(save_dir, save_file))
        print(traces['ipsi_movement'].shape, traces['contra_movement'].shape)
        if traces['ipsi_movement'].shape[1] >= 20 or traces['contra_movement'].shape[1] >= 20:
            contra_movement_traces[index, :] = np.mean(traces['contra_movement'], axis=1)
            ipsi_movement_traces[index, :] = np.mean(traces['ipsi_movement'], axis=1)
            i += 1
        else:
            contra_movement_traces[index, :] = np.nan
            ipsi_movement_traces[index, :] = np.nan
    all_mice_ipsi_traces[mouse_num, :] = np.nanmean(ipsi_movement_traces, axis=0)
    all_mice_contra_traces[mouse_num, :] = np.nanmean(contra_movement_traces, axis=0)




window_to_plot = [-0.5, 2]
inds_to_plot = np.where(np.logical_and(time_points>=window_to_plot[0], time_points<=window_to_plot[1]))
time_points_for_plot = decimate(time_points[inds_to_plot], 10)


contra_mean_trace = np.nanmean(all_mice_contra_traces, axis=0)[inds_to_plot]
ipsi_mean_trace = np.nanmean(all_mice_ipsi_traces, axis=0)[inds_to_plot]


set_plotting_defaults()

contra_mean_trace = decimate(np.nanmean(all_mice_contra_traces, axis=0)[inds_to_plot], 10)
ipsi_mean_trace = decimate(np.nanmean(all_mice_ipsi_traces, axis=0)[inds_to_plot], 10)
fig, axs = plt.subplots(1, 1, figsize=[2.5, 2])

axs.plot(time_points_for_plot, contra_mean_trace, color='#002F3A')
axs.plot(time_points_for_plot, ipsi_mean_trace, color='#76A8DA')

contra_error_bar_lower, contra_error_bar_upper = calculate_error_bars(contra_mean_trace,
                                                                decimate(all_mice_contra_traces[:, inds_to_plot[0]],10) ,
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, contra_error_bar_lower, contra_error_bar_upper, alpha=0.5,
                            facecolor='#002F3A', linewidth=0)

ipsi_error_bar_lower, ipsi_error_bar_upper = calculate_error_bars(ipsi_mean_trace,
                                                                decimate(all_mice_ipsi_traces[:, inds_to_plot[0]], 10),
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, ipsi_error_bar_lower, ipsi_error_bar_upper, alpha=0.5,
                            facecolor='#76A8DA', linewidth=0)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.yaxis.set_ticks_position('left')
axs.xaxis.set_ticks_position('bottom')

axs.axvline(0, color='k')
axs.set_ylabel('z-scored fluorescence')
axs.set_xlabel('time from movement start (s)')
plt.tight_layout()
axs.set_xlim(window_to_plot)

figure_dir = r'T:\paper\revisions\return to centre'
plt.tight_layout()
#plt.savefig(os.path.join(figure_dir, 'return_to_centre_average_traces_all_mice_min20trials.pdf'))
#plt.savefig(os.path.join(figure_dir, 'return_to_centre_average_traces_all_mice_min_20_trials.svg'))


# Example mouse

mouse = 'SNL_photo21'
num_sessions = 3
site = 'tail'

time_points = np.linspace(-5, 5, 100000)

experiments = experiments_to_process[experiments_to_process['mouse_id'] == mouse].reset_index(drop=True)
contra_movement_traces = np.zeros([num_sessions, 100000])
ipsi_movement_traces = np.zeros([num_sessions, 100000])
for index, experiment in experiments.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    save_dir = 'T:\\photometry_2AC\\processed_data\\return_to_centre\\{}'.format(mouse)
    save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_300frame_window.npz'.format(mouse, date)
    traces = np.load(os.path.join(save_dir, save_file))
    print(traces['ipsi_movement'].shape, traces['contra_movement'].shape)
    if traces['ipsi_movement'].shape[1] >= 20 or traces['contra_movement'].shape[1] >= 20:
        contra_movement_traces[index, :] = np.mean(traces['contra_movement'], axis=1)
        ipsi_movement_traces[index, :] = np.mean(traces['ipsi_movement'], axis=1)
    else:
        contra_movement_traces[index, :] = np.nan
        ipsi_movement_traces[index, :] = np.nan



fig, axs = plt.subplots(1, 1, figsize=[2.5, 2])
ipsi_mean_trace = decimate(np.nanmean(ipsi_movement_traces, axis=0)[inds_to_plot],10)
contra_mean_trace = decimate(np.nanmean(contra_movement_traces, axis=0)[inds_to_plot],10)
axs.plot(time_points_for_plot, contra_mean_trace, color='#002F3A')
axs.plot(time_points_for_plot, ipsi_mean_trace, color='#76A8DA')

contra_error_bar_lower, contra_error_bar_upper = calculate_error_bars(contra_mean_trace,
                                                                decimate(contra_movement_traces[:, inds_to_plot[0]],10) ,
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, contra_error_bar_lower, contra_error_bar_upper, alpha=0.5,
                            facecolor='#002F3A', linewidth=0)

ipsi_error_bar_lower, ipsi_error_bar_upper = calculate_error_bars(ipsi_mean_trace,
                                                                decimate(ipsi_movement_traces[:, inds_to_plot[0]], 10),
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, ipsi_error_bar_lower, ipsi_error_bar_upper, alpha=0.5,
                            facecolor='#76A8DA', linewidth=0)


axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.yaxis.set_ticks_position('left')
axs.xaxis.set_ticks_position('bottom')

axs.axvline(0, color='k')
axs.set_ylabel('z-scored fluorescence')
axs.set_xlabel('time from movement start (s)')
plt.tight_layout()
axs.set_xlim(window_to_plot)

figure_dir = r'T:\paper\revisions\return to centre'
plt.tight_layout()
#plt.savefig(os.path.join(figure_dir, 'return_to_centre_average_traces_example_min20trials.pdf'))
#plt.savefig(os.path.join(figure_dir, 'return_to_centre_average_traces_example_min_20_trials.svg'))

plt.show()





