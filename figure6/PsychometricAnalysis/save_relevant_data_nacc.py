"""In this module, we save only the relevant data from the tail dataframes to a new csv file.
"""

import pandas as pd

# Set up data directory
data_dir = '/Users/jessegeerts/Projects/CescaPsychometricAnalysis/psychometric_data'

# Load data
nacc_file = 'all_trial_data_Nacc_contra_ipsi_last_trial_confidence_and_traces_no_tracking_reward_aligned_pk5.pkl'
print('Loading data for nacc...')
nacc_data = pd.read_pickle(data_dir + '/' + nacc_file)[['mouse', 'session', 'fiber side', 'trial numbers', 'trial type', 'side', 'outcome', 'last trial type', 'last choice', 'last outcome', 'next trial type', 'next choice', 'next outcome', 'norm APE', 'stay or switch']].dropna().reset_index(drop=True)

print('Saving only necessary data to new pickle...')
# nacc_data.to_pickle(data_dir + '/' + 'nacc_data.pkl')
nacc_data.to_csv(data_dir + '/' + 'nacc_data.csv')