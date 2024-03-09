from utils.kernel_regression.linear_regression_utils import *
from utils.kernel_regression.return_to_centre_regression_utils import run_regression_return_to_centre_one_mouse_one_session, get_first_x_sessions_reg_rtc
from utils.post_processing_utils import remove_exps_after_manipulations
from set_global_params import processed_data_path, experiment_record_path, mice_average_traces
import gc
import os

"""
this is the same as in fig 3 but one mouse there is no tracking for one session so this reruns the regression 
with only the sessions we have tracking for (to make a fair comparison) 
"""
site = 'tail'
mouse_ids = mice_average_traces[site]
num_sessions = 3

experiment_record = pd.read_csv(experiment_record_path, dtype='str')
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
all_experiments_to_process = all_experiments_to_process[(all_experiments_to_process['include return to centre'] != 'no') & (all_experiments_to_process['include'] != 'no')].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process, x=num_sessions).reset_index(
    drop=True)
var_exps = []
for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    var_exp = run_regression_return_to_centre_one_mouse_one_session(mouse, date, reg_type='_no_return_to_centre')
    var_exps.append(var_exp)
    gc.collect()
experiments_to_process['var exp'] = var_exps
var_exp_filename = os.path.join(processed_data_path, '_'.join(mouse_ids) + 'var_exp_without_return_to_centre.p')
with open(var_exp_filename, "wb") as f:
    pickle.dump(experiments_to_process, f)