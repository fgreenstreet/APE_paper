import sys
import nptdms
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
import freely_moving_photometry_analysis.data_preprocessing.bpod_data_processing as bpod
import numpy as np


def get_camera_trigger_times(mouse, date, protocol):
    daq_file = bpod.find_daq_file(mouse, date)
    data = nptdms.TdmsFile(daq_file)
    sampling_rate = 10000

    main_session_file = bpod.find_bpod_file(mouse, date, protocol)
    loaded_bpod_file, trial_raw_events = bpod.load_bpod_file(main_session_file)

    clock = data.group_channels('acq_task')[3].data
    stim_trigger = data.group_channels('acq_task')[4].data
    stim_trigger_gaps = np.diff(stim_trigger)
    trial_start_ttls_daq_samples = np.where(stim_trigger_gaps > 2.595)
    trial_start_ttls_daq = trial_start_ttls_daq_samples[0] / sampling_rate
    daq_num_trials = trial_start_ttls_daq.shape[0]
    bpod_num_trials = trial_raw_events.shape[0]
    if daq_num_trials != bpod_num_trials:
        print('numbers of trials do not match!')
        print('daq: ', daq_num_trials)
        print('bpod: ', bpod_num_trials)
    else:
        print(daq_num_trials, 'trials in session')

    clock_diff = np.diff(clock)
    inds = np.where(clock > 5)[0] - 1
    first_high_inds = np.where(clock[inds]< 5)[0]
    camera_triggers = inds[first_high_inds]
    return camera_triggers, trial_start_ttls_daq_samples[0]


def find_nearest_trials(target_trials, other_trials):
     differences = (target_trials.reshape(1, -1) - other_trials.reshape(-1, 1))
     differences[np.where(differences < 0)] = 0
     indices = np.abs(differences).argmin(axis=0)
     return indices