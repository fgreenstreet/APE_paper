import os
from utils.tracking_analysis.fede_load_tracking import prepare_tracking_data
from utils.tracking_analysis.dlc_processing_utils import get_raw_photometry_data
from utils.post_processing_utils import remove_exps_after_manipulations
from utils.kernel_regression.linear_regression_utils import rolling_zscore
from utils.kernel_regression.return_to_centre_regression_utils import get_first_x_sessions_reg_rtc
from utils.return_to_centre_utils import *

mouse_ids = ['SNL_photo57', 'SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
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
                                                                                                     port_coords,
                                                                                                     photometry_data,
                                                                                                     side=ipsi_port,
                                                                                                     side_port=fiber_side,
                                                                                                     time_frame=300,
                                                                                                     short_turns_only=short_turns)

    ipsi_movement_traces, ipsi_movement_onsets_time_stamps = get_photometry_for_one_side_returns(contra_fiber_side_numeric,
                                                                                                     camera_triggers,
                                                                                                     trial_data,
                                                                                                     tracking_data,
                                                                                                     contra_turn_ang_thresh,
                                                                                                     port_coords,
                                                                                                     photometry_data,
                                                                                                     side=contra_port,
                                                                                                     side_port=fiber_options[np.where(fiber_options != fiber_side)],
                                                                                                     time_frame=300,
                                                                                                     short_turns_only=short_turns)

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