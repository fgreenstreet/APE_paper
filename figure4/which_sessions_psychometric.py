import pandas as pd
from set_global_params import bias_analysis_mice, experiment_record_path
from utils.psychometric_post_processing_utils import get_first_x_sessions


def get_psycho_sessions(mice, site, num_sessions, experiment_record):
    clean_experiments = experiment_record[(experiment_record['experiment_notes'] == 'psychometric')].reset_index(
        drop=True)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)
    all_experiments_to_process = all_experiments_to_process[all_experiments_to_process['include'] != 'no'].reset_index \
        (drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process, x=num_sessions).reset_index(
        drop=True)
    return experiments_to_process


site = 'Nacc'
mouse_ids = bias_analysis_mice[site]
save = False
num_sessions = 2

experiment_record = pd.read_csv(experiment_record_path, dtype='str')
experiment_record['date'] = experiment_record['date'].astype(str)
nacc_exps = get_psycho_sessions(mouse_ids, site, num_sessions, experiment_record)

# first round of tail mice
site = 'tail'
mouse_ids = bias_analysis_mice[site][0]
num_sessions = 3
tail1_exps = get_psycho_sessions(mouse_ids, site, num_sessions, experiment_record)

# this is for the second round of tail psychometric mice
mouse_ids = bias_analysis_mice[site][1]
num_sessions = 4
tail2_exps = get_psycho_sessions(mouse_ids, site, num_sessions, experiment_record)
all_psycho_exps = pd.concat([nacc_exps, tail1_exps, tail2_exps])
all_psycho_exps.to_csv('S:\projects\APE_data_francesca_for_paper\psychometric_exps.csv')
