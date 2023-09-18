"""In this module, we save only the relevant data from the tail dataframes to a new pickle file.
"""

import pandas as pd

# Set up data directory
data_dir = '/Users/jessegeerts/Projects/CescaPsychometricAnalysis/psychometric_data'

# Load data
tail_file_old_mice = 'all_trial_data_tail_contra_ipsi_last_trial_confidence_and_traces_no_tracking_choice_aligned_old_data_pk5.pkl'
tail_file_new_mice = 'all_trial_data_tail_contra_ipsi_last_trial_confidence_and_traces_no_tracking_choice_aligned_pk5.pkl'
print('Loading data for old mice...')
tail_old_mouse_data = pd.read_pickle(data_dir + '/' + tail_file_old_mice)[['mouse', 'session', 'fiber side', 'trial numbers', 'trial type', 'side', 'outcome', 'last trial type', 'last choice', 'last outcome', 'next trial type', 'next choice', 'next outcome', 'norm APE', 'stay or switch']].dropna().reset_index(drop=True)
print('Loading data for new mice...')
tail_new_mouse_data = pd.read_pickle(data_dir + '/' + tail_file_new_mice)[['mouse', 'session', 'fiber side', 'trial numbers', 'trial type', 'side', 'outcome', 'last trial type', 'last choice', 'last outcome', 'next trial type', 'next choice', 'next outcome', 'norm APE', 'stay or switch']].dropna().reset_index(drop=True)
all_tail_data = pd.concat([tail_old_mouse_data, tail_new_mouse_data]).reset_index(drop=True)

print('Saving only necessary data to new pickle...')
# all_tail_data.to_pickle(data_dir + '/' + 'all_tail_data.pkl')
all_tail_data.to_csv(data_dir + '/' + 'all_tail_data.csv')
