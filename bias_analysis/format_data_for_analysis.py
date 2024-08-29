from scipy import stats
import os
import sys
sys.path.append('..\..')
import seaborn as sns
from utils.tracking_analysis.tracking_plotting import *
from set_global_params import processed_data_path, bias_path, change_over_time_mice
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, remove_manipulation_days
from utils.tracking_analysis.extract_movement_for_all_sessions_utils import get_actual_trial_numbers, get_photometry_data


def get_session_with_10000th_trial(mouse, experiments):
    dates = experiments[experiments['mouse_id']==mouse]['date'].unique()
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    if session_starts[-1] >= 10000:
        last_session_idx = np.where(np.asarray(session_starts) >=10000)[0][0]
        print(np.asarray(session_starts) >=10000)
    else:
        last_session_idx = -1
        print(session_starts[-1])
    last_session_date = dates[last_session_idx]
    return(last_session_date)


recording_site = 'Nacc'
for mouse in change_over_time_mice[recording_site]:
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all_experiments = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    last_session = get_session_with_10000th_trial(mouse, experiments_to_process)

    dates = experiments_to_process['date'].values
    last_session_ind = int(np.where(dates == last_session)[0])
    for i, date in enumerate(dates[0:last_session_ind + 1]):
            photometry_data = get_photometry_data(mouse, date)
            saving_folder = os.path.join(processed_data_path, mouse)
            restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
            trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))

            # Initialize empty lists to collect data
            trial_nums = []
            reaction_times = []
            peaks = []
            trial_types = []

            if recording_site == 'tail':
                aligned_data = photometry_data.choice_data
            else:
                aligned_data = photometry_data.cue_data
            # Single loop to iterate over trial types and collect data
            for trial_type in ['contra_data', 'ipsi_data']:
                trial_type_data = getattr(aligned_data, trial_type)

                # Extend lists with data from the current trial type
                trial_nums.extend(trial_type_data.trial_nums)
                reaction_times.extend(trial_type_data.reaction_times)
                peaks.extend(trial_type_data.trial_peaks)

                # Add the trial type for each entry
                trial_types.extend([trial_type] * len(trial_type_data.trial_nums))

            # Create the DataFrame
            df = pd.DataFrame({
                'trial_nums': trial_nums,
                'reaction_times': reaction_times,
                'APE_peaks': peaks,
                'trial_type': trial_types
            })


            df_sorted = df.sort_values(by='trial_nums')

            # Assuming df_sorted is the sorted DataFrame with 'trial_nums', 'reaction_times', 'peaks', 'trial_type'
            # Assuming trial_data is the DataFrame with many columns, one of which is 'Trial num'

            # First, rename the 'Trial num' column in trial_data to match 'trial_nums' in df_sorted for merging
            trial_data = trial_data.rename(columns={'Trial num': 'trial_nums'})

            # Merge the two DataFrames on 'trial_nums', using a left join to keep all rows in trial_data
            merged_df = pd.merge(trial_data, df_sorted, on='trial_nums', how='left')
            merged_df['actual trial nums'] = get_actual_trial_numbers(merged_df['trial_nums'], date, mouse, recording_site=recording_site)
            if i == 0:
                all_session_data = merged_df
            else:
                all_session_data = pd.concat([all_session_data, merged_df])

    all_session_data = all_session_data.reset_index(drop=True)
    if not os.path.exists(bias_path):
        os.makedirs(bias_path)
    bias_file = os.path.join(bias_path, 'pre_processing_bias_{}.pkl'.format(mouse))
    all_session_data.to_pickle(bias_file)


