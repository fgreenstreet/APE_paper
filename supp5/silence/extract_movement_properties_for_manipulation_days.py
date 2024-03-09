import numpy as np
from scipy.signal import filtfilt
from scipy.optimize import curve_fit
import pickle
import numpy as np
import pandas as pd
import os
from fede_load_tracking import prepare_tracking_data
from camera_trigger_preprocessing_utils import *
from plotting import *
from dlc_processing_utils import get_photometry_data
from velocity_utils import format_tracking_data_and_photometry, format_only_photometry
from extract_movement_for_all_sessions_utils import *

sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git\\freely_moving_photometry_analysis')

from freely_moving_photometry_analysis.utils.reaction_time_utils import get_bpod_trial_nums_per_session
from freely_moving_photometry_analysis.utils.post_processing_utils import get_all_experimental_records
from freely_moving_photometry_analysis.utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings, remove_manipulation_days
#TODO: needs massive work! Just came from DLC_repo
#mouse = 'SNL_photo21'
mice = ['SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo37', 'SNL_photo43']
recording_site = 'tail'
load_saved = False
for mouse in mice:
    all_experiments = get_all_experimental_records()
    all_experiments = remove_bad_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site) & (all_experiments['experiment_notes'] == 'state change white noise')]   #'state change white noise' #'silence'
    dates = experiments_to_process['date'].values[-4:]

    for date in dates:
        save_out_folder = 'S:\\projects\\APE_tracking\\{}\\{}\\'.format(mouse, date)
        #save_out_folder = 'C:\\Users\\francescag\\Documents\\PhD_Project\\photometry_2AC_dlc\\temp_whilst_ceph_is_down\\{}\\{}\\'.format(mouse, date)
        if not os.path.exists(save_out_folder):
            os.makedirs(save_out_folder)
        movement_param_file = os.path.join(save_out_folder, 'APE_tracking{}_{}.pkl'.format(mouse, date)) #'APE_tracking_also_incorrect{}_{}.pkl'.format(mouse, date))
        if not os.path.isfile(movement_param_file) & (load_saved):
            quantile_data, trial_data = get_movement_properties_for_session(mouse, date, protocol='State_Change_Two_Alternative_Choice', multi_session=False) #'Two_Alternative_Choice_Inference' #'Two_Alternative_Choice' #State_Change_Two_Alternative_Choice'
            quantile_data.to_pickle(movement_param_file)