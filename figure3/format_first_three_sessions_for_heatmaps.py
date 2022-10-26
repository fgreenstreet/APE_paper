import pandas as pd
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings, get_first_x_sessions, get_all_experimental_records, add_experiment_to_aligned_data
from set_global_params import experiment_record_path

if __name__ == '__main__':
    mouse_ids = ['SNL_photo57', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
    site = 'tail'
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process)
    add_experiment_to_aligned_data(experiments_to_process)
