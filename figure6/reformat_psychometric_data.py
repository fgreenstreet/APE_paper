import os
from set_global_params import bias_analysis_mice, processed_data_path
from utils.psychometric_post_processing_utils import get_all_psychometric_session_dlc

# this came from a different repo so check imports
#   TODO: sort out imports
#   TODO: remove tracking analysis from this


site = 'Nacc'

mouse_ids = bias_analysis_mice[site]

save = False
num_sessions = 2
key = 'fitted max cumsum ang vel'
alignment = 'reward'
all_trial_data = get_all_psychometric_session_dlc(mouse_ids, site, save=False, load_saved=False, key=key, get_movement=False, align_to=alignment, num_sessions=num_sessions)
save_path = os.path.join(processed_data_path, 'psychometric_data')

all_trial_data.to_pickle(os.path.join(save_path,"all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces_no_tracking_{}_aligned_pk5.pkl".format(site, alignment)))
# trial type, 'next trial type', 'last trial type, 'mouse', 'recording site', 'norm APE', 'side', 'fiber side'