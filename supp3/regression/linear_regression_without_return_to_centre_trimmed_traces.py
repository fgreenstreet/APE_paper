import os
from utils.kernel_regression.linear_regression_utils import *
import gc
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.kernel_regression.return_to_centre_regression_utils import get_first_x_sessions_reg_rtc, run_regression_one_mouse_one_session_no_return_no_trim

mouse_ids = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo57', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72'] #'SNL_photo57', 'SNL_photo16', 'SNL_photo17', 'SNL_photo18',
site = 'tail'
experiment_record_path = 'W:\\photometry_2AC\\experimental_record.csv'
processed_data_path = 'W:\\photometry_2AC\\processed_data\\'

experiment_record = pd.read_csv(experiment_record_path)
experiment_record['date'] = experiment_record['date'].astype(str)
good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
clean_experiments = remove_bad_recordings(good_experiments)
all_experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process)
var_exps = []
scores = []
for index, experiment in experiments_to_process.iterrows():
    score, var_exp = run_regression_one_mouse_one_session_no_return_no_trim(experiment)
    var_exps.append(var_exp)
    scores.append(score)
file_name = site + '_explained_variances_all_cues_trimmed_traces_only_tracking_mice.p' #'_explained_variances_not_cleaned.p' #'_explained_variances.p'
processed_data_dir = os.path.join('T:\\photometry_2AC\\processed_data\\linear_regression_data\\')
saving_filename = os.path.join('T:\\photometry_2AC\\processed_data\\linear_regression_data\\', file_name)
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

regression_stats = experiments_to_process[['mouse_id', 'date']].reset_index(drop=True)
regression_stats['full model explained variance with trimming'] = var_exps
regression_stats['full model explained variance without trimming'] = scores

with open(saving_filename, "wb") as f:
    pickle.dump(regression_stats, f)

    gc.collect()

