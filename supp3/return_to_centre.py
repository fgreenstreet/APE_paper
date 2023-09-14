import sys

sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
import os
import pandas as pd
import dlc_processing_utils
from camera_trigger_preprocessing_utils import *
import seaborn as sns
from fede_geometry import r2, calc_angle_between_vectors_of_points_2d, calc_ang_velocity
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
from freely_moving_photometry_analysis.utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from freely_moving_photometry_analysis.utils.regression.linear_regression_utils import get_first_x_sessions
from freely_moving_photometry_analysis.utils.regression.regression_plotting_utils import make_box_plot
from freely_moving_photometry_analysis.utils.individual_trial_analysis_utils import get_photometry_around_event
import numpy as np
from scipy.signal import filtfilt
from scipy.optimize import curve_fit

import numpy as np
from fede_load_tracking import prepare_tracking_data
from velocity_utils import calculate_velocity
from camera_trigger_preprocessing_utils import *
from plotting import *
from numpy.linalg import norm
from dlc_processing_utils import get_raw_photometry_data

from velocity_utils import format_tracking_data_and_photometry, format_only_photometry

import pandas as pd
from camera_trigger_preprocessing_utils import *
from freely_moving_photometry_analysis.utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from freely_moving_photometry_analysis.utils.regression.linear_regression_utils import rolling_zscore
from freely_moving_photometry_analysis.utils.regression.return_to_centre_regression_utils import get_first_x_sessions_reg_rtc
from freely_moving_photometry_analysis.utils.regression.regression_plotting_utils import make_box_plot


def get_return_to_centre_window(trials, all_trials, all_camera_triggers):
    choice = (trials[(trials['State name'] == 'WaitForResponse') & (
            trials['Instance in state'] == trials['Max times in state'])]['Time end'].values * 10000)
    choice = choice.astype(int)
    choice_triggers = find_nearest_trials(choice, all_camera_triggers)
    trial_nums = trials['Trial num'].values
    next_trial_nums = trial_nums + 1
    max_trial_num = np.max(all_trials['Trial num'].values)
    valid_indices = np.where(next_trial_nums <= max_trial_num)
    next_trial_nums = next_trial_nums[valid_indices]
    choice_triggers = choice_triggers[valid_indices]
    next_trials = all_trials[all_trials['Trial num'].isin(next_trial_nums)]
    next_trial_starts = (next_trials[(next_trials['State name'] == 'WaitForPoke') & (
            next_trials['Instance in state'] == 1)]['Time end'].values * 10000)
    next_trial_starts = next_trial_starts.astype(int)
    next_trial_start_triggers = find_nearest_trials(next_trial_starts, all_camera_triggers)
    return choice_triggers, next_trial_start_triggers, choice


def get_initial_heading_vector(tracking_data, onset_times, offset_times, time_frame=15):
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, onset_times.shape[0]))
    heading_vectors = []
    body_parts = ['nose', 'left ear', 'right ear']
    fig, axs = plt.subplots(1, 2)
    for trial_num, start_time in enumerate(onset_times):
        initial_tracking = {}
        for body_part in body_parts:
            trial_tracking_x = tracking_data[body_part]['x'][start_time: offset_times[trial_num]]
            trial_tracking_y = tracking_data[body_part]['y'][start_time: offset_times[trial_num]]
            initial_tracking_body_part = pd.DataFrame({'x': trial_tracking_x[:time_frame].values, 'y': trial_tracking_y[:time_frame].values})
            initial_tracking[body_part] = initial_tracking_body_part
        vectors_in_time = np.vstack([np.diff(initial_tracking['nose']['x']), np.diff(initial_tracking['nose']['y'])])
        initial_vector = np.sum(vectors_in_time, axis=1)
        heading_vectors.append(initial_vector)

        #axs[0].plot(initial_tracking['nose']['x'], initial_tracking['nose']['y'], color=colours[trial_num])
        #axs[1].plot([0, initial_vector[0]], [0, initial_vector[1]], color=colours[trial_num])

    return heading_vectors, axs


def get_turn_start_times(tracking_data, onset_times, offset_times, turn_ang_thresh,  time_frame=40, side_port='left', short_turns_only=True):
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, onset_times.shape[0]))
    heading_vectors = []
    body_parts = ['nose', 'left ear', 'right ear']
    #fig, axs = plt.subplots(1, 4)
    #fig1, axs1 = plt.subplots(1, 4)
    cumsum_ang_vs = []
    turn_onsets = []
    for trial_num, start_time in enumerate(onset_times):
        initial_tracking = {}
        for body_part in body_parts:
            trial_tracking_x = tracking_data[body_part]['x'][start_time: offset_times[trial_num]]
            trial_tracking_y = tracking_data[body_part]['y'][start_time: offset_times[trial_num]]
            initial_tracking_body_part = pd.DataFrame(
                {'x': trial_tracking_x[:time_frame].values, 'y': trial_tracking_y[:time_frame].values})
            initial_tracking[body_part] = initial_tracking_body_part
        midpoints_y = (initial_tracking['left ear']['y'] + initial_tracking['right ear']['y']) / 2
        midpoints_x = (initial_tracking['left ear']['x'] + initial_tracking['right ear']['x']) / 2
        head_angles = calc_angle_between_vectors_of_points_2d(initial_tracking['nose']['x'].values,
                                                              initial_tracking['nose']['y'].values, midpoints_x,
                                                              midpoints_y)
        head_angular_velocity = calc_ang_velocity(head_angles)
        cumsum_ang_v = np.cumsum(head_angular_velocity)

        #axs1[0].plot(cumsum_ang_v, color=colours[trial_num])
        #axs1[1].plot(head_angular_velocity, color=colours[trial_num])
        #axs1[2].plot(head_angles)
        #axs1[0].axhline(turn_ang_thresh)


        if side_port == 'left':
            #turn_ang_thresh = 85
            if np.any(cumsum_ang_v <= turn_ang_thresh):
                turn_onset_ind = np.where(cumsum_ang_v <= turn_ang_thresh)[0][0]
                makes_full_turn = np.any(cumsum_ang_v[turn_onset_ind: turn_onset_ind + 50] <= turn_ang_thresh - 90)
                crosses_threshold = np.all(cumsum_ang_v[turn_onset_ind:] <= turn_ang_thresh)
                if short_turns_only:
                    short_duration = cumsum_ang_v[turn_onset_ind:].shape[0] <= 30
                else:
                    short_duration = True
                if makes_full_turn & crosses_threshold & short_duration:
                    turn_onsets.append(turn_onset_ind * 1 / 30)
                    #axs[0].plot(cumsum_ang_v[turn_onset_ind:], color=colours[trial_num])
                    #axs[1].plot(head_angular_velocity[turn_onset_ind:], color=colours[trial_num])
                    #axs[2].plot(head_angles[turn_onset_ind:])
                else:
                    turn_onsets.append(False)
            else:
                turn_onsets.append(False)
        else:
            #turn_ang_thresh = 280
            if np.any(cumsum_ang_v >= turn_ang_thresh):
                turn_onset_ind = np.where(cumsum_ang_v >= turn_ang_thresh)[0][0]
                makes_full_turn = np.any(cumsum_ang_v[turn_onset_ind: turn_onset_ind+50]>=turn_ang_thresh + 90)
                crosses_threshold = np.all(cumsum_ang_v[turn_onset_ind:]>=turn_ang_thresh)
                if short_turns_only:
                    short_duration = cumsum_ang_v[turn_onset_ind:].shape[0] <= 30
                else:
                    short_duration = True
                if makes_full_turn & crosses_threshold & short_duration:
                    turn_onsets.append(turn_onset_ind * 1 / 30)
                    #axs[0].plot(cumsum_ang_v[turn_onset_ind:], color=colours[trial_num])
                    #axs[1].plot(head_angular_velocity[turn_onset_ind:], color=colours[trial_num])
                    #axs[2].plot(head_angles[turn_onset_ind:])
                else:
                    turn_onsets.append(False)
            else:
                turn_onsets.append(False)
    # turn_onsets are in seconds
    return cumsum_ang_vs, turn_onsets


def get_movement_start_times(tracking_data, onset_times, offset_times, body_part='nose'):
    fig, axs = plt.subplots(1, 3)
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, len(onset_times)))
    velocities = []
    accelerations = []
    movement_onsets = []
    for trial_num, start_time in enumerate(onset_times):
        trial_tracking_x = tracking_data[body_part]['x'][start_time: offset_times[trial_num]]
        trial_tracking_y = tracking_data[body_part]['y'][start_time: offset_times[trial_num]]
        velocity = calculate_velocity(zip(trial_tracking_x, trial_tracking_y))
        acceleration = np.diff(velocity)
        velocities.append(velocity)
        accelerations.append(accelerations)
        axs[1].plot(acceleration, color=colours[trial_num], alpha=0.5)
        axs[0].plot(velocity, color=colours[trial_num], alpha=0.5)
        axs[2].plot(np.cumsum(velocity), color=colours[trial_num], alpha=0.5)
        axs[2].axhline(40)
        #movement_onsets.append(np.where(np.cumsum(velocity) >= 40)[0][0] * 1/30)
        if np.any(np.cumsum(acceleration) >= 15):
            movement_onsets.append(np.where(np.cumsum(acceleration) >= 15)[0][0] * 1 / 30)
        else:
            movement_onsets.append(False)
    return movement_onsets


def get_photometery_return_to_centre_certain_trials(trial_indices, all_trials, photometry_data):
    event_times = all_trials['Time end'].values[trial_indices]
    traces = get_photometry_around_event(event_times, photometry_data)
    norm_traces = stats.zscore(traces.T, axis=0)
    plt.figure()
    plt.plot(np.mean(norm_traces, axis=1))
    return norm_traces


def get_photometery_return_to_centre_certain_trials_aligned_to_movement(trial_indices, all_trials, movement_onset_delays, photometry_data):
    valid_trials = np.where(np.array(movement_onset_delays) != False)[0]
    trial_indices = trial_indices[valid_trials]
    movement_onset_delays = np.array(movement_onset_delays)[valid_trials]
    event_times = all_trials['Time end'].values[trial_indices] + movement_onset_delays
    traces = get_photometry_around_event(event_times, photometry_data)
    norm_traces = traces.T #stats.zscore(traces.T, axis=0)
    plt.figure()
    plt.plot(np.mean(norm_traces, axis=1))
    return norm_traces


def calculate_cosine_similarity_to_goal_vector(port_coords, heading_vectors, axs, side='side1'):
    similarities = []
    goal_vector = port_coords['centre'] - port_coords[side]
    for heading_vector in heading_vectors:
        similarities.append(calculate_cosine_similarity(heading_vector, goal_vector))
    plt.plot([0, goal_vector[0]], [0, goal_vector[1]], color='k', lw=4)
    axs[0].scatter([port_coords[side][0], port_coords['centre'][0]], [port_coords[side][1], port_coords['centre'][1]], color='k')
    return similarities, goal_vector


def calculate_cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def plot_direct_trials(similarities, heading_vectors, goal_vector, threshold=0.6):
    fig, axs = plt.subplots(1,1)
    axs.hist(similarities, bins=20)
    axs.axvline(threshold)
    trial_indices = np.where(np.array(similarities) >= threshold)[0]
    direct_vectors = np.array(heading_vectors)[trial_indices]
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, len(direct_vectors)))
    fig, axs = plt.subplots(1, 2)
    for trial_num, vector in enumerate(direct_vectors):
        axs[0].plot([0, vector[0]], [0, vector[1]], color=colours[trial_num])
    axs[0].plot([0, goal_vector[0]], [0, goal_vector[1]], color='k', lw=4)

    trial_indices = np.where(np.array(similarities) < threshold)[0]
    direct_vectors = np.array(heading_vectors)[trial_indices]
    colourmap = matplotlib.cm.inferno
    colours = colourmap(np.linspace(0, 0.8, len(direct_vectors)))
    for trial_num, vector in enumerate(direct_vectors):
        axs[1].plot([0, vector[0]], [0, vector[1]], color=colours[trial_num])
    axs[1].plot([0, goal_vector[0]], [0, goal_vector[1]], color='k', lw=4)


def get_photometry_for_one_side_returns(fiber_side_numeric, camera_triggers, trial_data, tracking_data, turn_ang_thresh, side='side1', side_port='left', time_frame=300, short_turns_only=True):
    trials = trial_data[(trial_data['State name'] == 'WaitForResponse') & (
            trial_data['Instance in state'] == trial_data['Max times in state']) & (
                                         trial_data['Response'] == fiber_side_numeric)]

    choice_triggers, next_trial_start_triggers, choice_times = get_return_to_centre_window(trials, trial_data, camera_triggers)

    heading_vectors, axs = get_initial_heading_vector(tracking_data, choice_triggers,
                                                      next_trial_start_triggers, time_frame=time_frame)

    cosine_similarities, goal_vector = calculate_cosine_similarity_to_goal_vector(port_coords, heading_vectors, axs,
                                                                                  side=side)
    plot_direct_trials(cosine_similarities, heading_vectors, goal_vector)
    trial_indices = np.where(np.array(cosine_similarities) >= 0.9)[0]

    # find movement onsets for direct trials
    cum_sum_ang_vs, turn_onsets = get_turn_start_times(tracking_data, choice_triggers[trial_indices],
                                                                 next_trial_start_triggers[trial_indices],
                                                                 turn_ang_thresh, side_port=side_port,
                                                                 time_frame=time_frame, short_turns_only=short_turns_only)
    contra_movement_inds = [i for i, t in enumerate(turn_onsets) if t]
    valid_choices = choice_times[trial_indices]
    contra_movement_onsets_time_stamps = np.array(valid_choices)[contra_movement_inds] / 10000 + \
                                         np.array(turn_onsets)[
                                             contra_movement_inds]  # these times are in seconds

    contra_movement_traces = get_photometery_return_to_centre_certain_trials_aligned_to_movement(trial_indices,
                                                                                                 trials,
                                                                                                 turn_onsets,
                                                                                                 photometry_data)
    return contra_movement_traces, contra_movement_onsets_time_stamps


mouse_ids = ['SNL_photo57']#['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
num_sessions = 3
site = 'tail'
use_old_tracking = False
short_turns = False


old_mice = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']
experiment_record = pd.read_csv('T:\\photometry_2AC\\experimental_record.csv', dtype=str)
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
all_experiments_to_process = all_experiments_to_process[all_experiments_to_process['include return to centre'] != 'no'].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process, x=num_sessions).reset_index(
    drop=True)

for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    save_out_folder = 'T:\\photometry_2AC\\tracking_analysis\\' + mouse

    port_coords = {'side1': np.array([int(experiments_to_process['left side x'][0]), int(experiments_to_process['left side y'][0])]),
                  'centre': np.array([int(experiments_to_process['centre port x'][0]), int(experiments_to_process['centre port y'][0])]),
                  'side2': np.array([int(experiments_to_process['right side x'][0]), int(experiments_to_process['right side y'][0])])}

    if mouse in old_mice and use_old_tracking:
        file_path = 'T:\\deeplabcut_tracking\\second_attempt_test_videos\\{}_{}DLC_resnet50_two_acMay10shuffle1_600000.h5'.format(mouse, date)
        protocol = 'Two_Alternative_Choice'
        left_ang_thresh = 85
        right_ang_thresh = 280

    else:
        file_path = 'S:\\projects\\APE_tracking\\{}\\{}\\cameraDLC_resnet50_train_network_with_more_miceMar2shuffle1_800000.h5'.format(
            mouse, date)
        protocol = 'Two_Alternative_Choice_CentrePortHold'
        left_ang_thresh = 100
        right_ang_thresh = 300
        if mouse in old_mice:
            protocol = 'Two_Alternative_Choice'
            left_ang_thresh = 85
            right_ang_thresh = 280



    body_parts = ('nose', 'left ear', 'right ear', 'tail base', 'tail tip')
    tracking_data = prepare_tracking_data(
        tracking_filepath=file_path,
        tracking=None,
        bodyparts=body_parts,
        likelihood_th=0.999,
        median_filter=True,
        filter_kwargs={},
        compute=True,
        smooth_dir_mvmt=True,
        interpolate_nans=True,
        verbose=False)

    camera_triggers, trial_start_stamps = get_camera_trigger_times(mouse, date, protocol)
    trial_start_triggers = find_nearest_trials(trial_start_stamps, camera_triggers)
    dff_data, trial_data = get_raw_photometry_data(mouse, date)
    photometry_data = rolling_zscore(pd.Series(dff_data), window=10 * 10000)

    fiber_side = experiment['fiber_side']
    if fiber_side == 'left':
        ipsi_port = 'side1'
        contra_port = 'side2'
        ipsi_turn_ang_thresh = left_ang_thresh
        contra_turn_ang_thresh = right_ang_thresh
    else:
        ipsi_port = 'side2'
        contra_port = 'side1'
        ipsi_turn_ang_thresh = right_ang_thresh
        contra_turn_ang_thresh = left_ang_thresh

    fiber_options = np.array(['left', 'right'])
    fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]
    contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]

    contra_movement_traces, contra_movement_onsets_time_stamps = get_photometry_for_one_side_returns(fiber_side_numeric,
                                                                                                     camera_triggers,
                                                                                                     trial_data,
                                                                                                     tracking_data,
                                                                                                     ipsi_turn_ang_thresh,
                                                                                                     side=ipsi_port,
                                                                                                     side_port=fiber_side,
                                                                                                     time_frame=300,
                                                                                                     short_turns_only=short_turns)

    ipsi_movement_traces, ipsi_movement_onsets_time_stamps = get_photometry_for_one_side_returns(contra_fiber_side_numeric,
                                                                                                     camera_triggers,
                                                                                                     trial_data,
                                                                                                     tracking_data,
                                                                                                     contra_turn_ang_thresh,
                                                                                                     side=contra_port,
                                                                                                     side_port=fiber_options[np.where(fiber_options != fiber_side)],
                                                                                                     time_frame=300,
                                                                                                     short_turns_only=short_turns)


    #plt.show()
    plt.close('all')
    save_dir = 'T:\\photometry_2AC\\processed_data\\return_to_centre\\{}'.format(mouse)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if short_turns:
        save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_300frame_window.npz'.format(mouse, date)
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times_300frame_window.npz'.format(mouse, date)
    else:
        save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_300frame_window_long_turns.npz'.format(
            mouse, date)
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times_300frame_window_long_turns.npz'.format(mouse, date)
    np.savez(os.path.join(save_dir, save_file), contra_movement=contra_movement_traces, ipsi_movement=ipsi_movement_traces)
    np.savez(os.path.join(save_dir, time_stamp_save_file), contra_movement_return=contra_movement_onsets_time_stamps,
             ipsi_movement_return=ipsi_movement_onsets_time_stamps)