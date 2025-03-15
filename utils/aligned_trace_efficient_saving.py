# load aligned trace
# create pickled object with almost all of the data but no traces (initialise these as None)
# save out separately npy files with traces aligned to: 1) choice, 2) cue, 3) reward
# 1) ipsi and contra
# 2) ipsi and contra
# 3) ipsi and contra

import os
import pickle
from set_global_params import processed_data_path
from copy import deepcopy
import numpy as np

mouse = 'SNL_photo16'
date = '20200210'
saving_folder = os.path.join(processed_data_path, mouse)
aligned_filename = os.path.join(saving_folder, mouse + '_' + date + '_' + 'aligned_traces.p')
with open(aligned_filename, 'rb') as f:
    temp_data = pickle.load(f)
data = deepcopy(temp_data)
ipsi_cue = data.cue_data.ipsi_data.sorted_traces
data.cue_data.ipsi_data.sorted_traces = None
contra_cue = data.cue_data.contra_data.sorted_traces
data.cue_data.contra_data.sorted_traces = None
ipsi_choice = data.choice_data.ipsi_data.sorted_traces
data.choice_data.ipsi_data.sorted_traces = None
contra_choice = data.choice_data.contra_data.sorted_traces
data.choice_data.contra_data.sorted_traces = None
ipsi_reward = data.reward_data.ipsi_data.sorted_traces
data.reward_data.ipsi_data.sorted_traces = None
contra_reward = data.reward_data.contra_data.sorted_traces
data.reward_data.contra_data.sorted_traces = None

trace_file_root = os.path.join(saving_folder, mouse + '_' + date + '_')
obj_file = os.path.join(saving_folder, mouse + '_' + date + '_' + 'aligned_traces_efficient_save.p')
#np.save(trace_file_root + 'ipsi_cue_traces.npy', ipsi_cue) # this file is still too big - I need to save a smaller window - it's already zscored so it should be ok?
with open(obj_file, 'wb') as f:
    pickle.dump(data, f)