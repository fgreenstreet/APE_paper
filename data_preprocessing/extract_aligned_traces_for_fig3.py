import pickle
import os
from utils.individual_trial_analysis_utils import SessionData
import pandas as pd
from utils.post_processing_utils import remove_exps_after_manipulations, get_first_x_sessions
from set_global_params import experiment_record_path
from set_global_params import processed_data_path


def add_experiment_to_aligned_data(experiments_to_add):
    """
    Gets raw data and re-formats it into an object that is easy to access for plotting later.
    Only needed for figure 3 as this id the only time this distinction is made
    Slightly different to version in extract_aligned_traces_for_fig_general.py (has incorrect outcome responses too)
    Args:
        experiments_to_add (pd.Dataframe): rows from experiment record

    Returns:

    """
    data_root = os.path.join(processed_data_path, 'for_figure')
    for index, experiment in experiments_to_add.iterrows():
        print(experiment['mouse_id'],' ', experiment['date'])
        saving_folder = os.path.join(data_root, experiment['mouse_id'])
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_traces.get_choice_responses()
        session_traces.get_cue_responses()
        session_traces.get_reward_responses()
        session_traces.get_outcome_responses()
        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'aligned_traces_for_fig.p'
        save_filename = os.path.join(saving_folder, aligned_filename)
        pickle.dump(session_traces, open(save_filename, "wb"))


if __name__ == '__main__':
    mouse_ids = ['SNL_photo57', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72'] # can run this on any mice
    site = 'tail'
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process)
    add_experiment_to_aligned_data(experiments_to_process)
