import numpy as np
import pickle
from utils.individual_trial_analysis_regression_all_cues import SessionEvents
import pandas as pd
import os
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.linear_regression_utils import get_first_x_sessions

from set_global_params import experiment_record_path, processed_data_path


def get_all_experimental_records():
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record


def add_timestamps_to_aligned_data(experiments_to_add):
    for index, experiment in experiments_to_add.iterrows():
        data_folder = processed_data_path + experiment['mouse_id'] + '\\'
        saving_folder = processed_data_path + experiment['mouse_id'] + '\\linear_regression\\'
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        #trial_data = pd.read_pickle(data_folder + restructured_data_filename)
        dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
        #dff = np.load(data_folder + dff_trace_filename)
        print(experiment['mouse_id'], experiment['date'])
        session_events = SessionEvents(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_events.get_choice_events()
        session_events.get_cue_events()
        session_events.get_reward_events()

        choice_trial_nums = np.sort(np.concatenate(
            [session_events.choice_data.contra_data.trial_nums, session_events.choice_data.ipsi_data.trial_nums]))
        cue_trial_nums = np.sort(np.concatenate(
            [session_events.cue_data.low_cue_data.trial_nums, session_events.cue_data.high_cue_data.trial_nums]))
        common_trial_nums = np.intersect1d(cue_trial_nums, choice_trial_nums)

        session_events.choice_data.contra_data.filter_by_trial_nums(common_trial_nums)
        session_events.choice_data.ipsi_data.filter_by_trial_nums(common_trial_nums)
        session_events.cue_data.high_cue_data.filter_by_trial_nums(common_trial_nums)
        session_events.cue_data.low_cue_data.filter_by_trial_nums(common_trial_nums)

        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'behavioural_events_with_no_rewards_all_cues_matched_trials.p' #'behavioural_events_clean_cues.p' #'behavioural_events_no_repeated_cues.p'
        save_filename = saving_folder + aligned_filename
        pickle.dump(session_events, open(save_filename, "wb"))


def remove_manipulation_days(experiments):
    exemption_list = ['psychometric', 'state change medium cloud', 'value blocks', 'state change white noise', 'omissions and large rewards']
    exemptions = '|'.join(exemption_list)
    index_to_remove = experiments[np.logical_xor(experiments['include'] == 'no', experiments['experiment_notes'].str.contains(exemptions, na=False))].index
    cleaned_experiments = experiments.drop(index=index_to_remove)
    return cleaned_experiments



if __name__ == '__main__':
    mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
    #['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo57', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
    #['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35'] #['SNL_photo57', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
        #['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
    site = 'Nacc'
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    clean_experiments = remove_bad_recordings(good_experiments)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process)
    add_timestamps_to_aligned_data(experiments_to_process)

    # for mouse_id in mouse_ids:
    #     date = '20201121'
    #     all_experiments = get_all_experimental_records()
    #     clean_experiments = remove_exps_after_manipulations(all_experiments, [mouse_id])
    #     site = 'Nacc'
    #     index_to_remove = clean_experiments[clean_experiments['recording_site'] != site].index
    #     cleaned_experiments = clean_experiments.drop(index=index_to_remove)
    #
    #
    #     if (mouse_id =='all') & (date == 'all'):
    #         experiments_to_process = cleaned_experiments
    #     elif (mouse_id == 'all') & (date != 'all'):
    #         experiments_to_process = cleaned_experiments[cleaned_experiments['date'] == date]
    #     elif (mouse_id != 'all') & (date == 'all'):
    #         experiments_to_process = cleaned_experiments[cleaned_experiments['mouse_id'] == mouse_id]
    #     elif (mouse_id != 'all') & (date != 'all'):
    #         experiments_to_process = cleaned_experiments[(cleaned_experiments['date'] == date) & (cleaned_experiments['mouse_id'] == mouse_id)]
    #     add_timestamps_to_aligned_data(experiments_to_process)

