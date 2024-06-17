# PATHS
experiment_record_path = 'T:\\photometry_2AC\\experimental_record.csv'
processed_data_path = 'T:\\photometry_2AC\\processed_data\\'
post_processed_tracking_data_path = 'T:\\photometry_2AC\\tracking_analysis\\'
behavioural_data_path = 'T:\\photometry_2AC\\bpod_data\\'
photometry_data_path = 'T:\\photometry_2AC\\freely_moving_photometry_data\\'
figure_directory = 'T:\\paper\\'
running_in_box_dir = 'T:\\photometry_2AC\\running_in_box_photometry\\'
running_in_box_tracking_dir = 'T:\\deeplabcut_running_in_box\\running_in_box\\'
psychometric_data_path = 'T:\\photometry_2AC\\tracking_analysis_no_tracking\\'
raw_tracking_path = 'S:\\projects\\APE_tracking\\'
old_raw_tracking_path = 'T:\\deeplabcut_tracking\\second_attempt_test_videos\\'

# CONSTANTS
daq_sample_rate = 10000  # Hz
camera_sample_rate = 30  # Hz

# MICE TO INCLUDE FOR FIGURES
# Figure 2
mice_average_traces = {'tail': ['SNL_photo17', 'SNL_photo16', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo57', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72'],
                       'Nacc': ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35'],
                       'tail_ant': ['SNL_photo14', 'SNL_photo31', 'SNL_photo32']}

# Figure 3
change_over_time_mice = {'tail': ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26'],
                         'Nacc': ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']}

beginning_and_end_comparison_mice = {'tail': ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo37', 'SNL_photo43', 'SNL_photo44'],
                                     'Nacc': ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']}
# beginning and end comparison uses more mice than change over time as change over time required continuous recording every few days so there are fewer mice

# Figure 4
fig4_plotting_colours = {'Nacc': ['#E95F32', '#f78b43', '#F9C0AF'], 'tail': ['#00343a', '#62b3c4', '#bcebee']}
# reward size change experiment
large_reward_omission_mice = {'tail': ['SNL_photo37', 'SNL_photo43', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26'],
                              'Nacc': ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo32', 'SNL_photo34', 'SNL_photo35']}
large_reward_omission_example_mice = {'Nacc': 'SNL_photo31', 'tail': 'SNL_photo26'}

# value change experiment (reward amount change in blocks)
value_change_mice = {'tail': ['SNL_photo70', 'SNL_photo72', 'SNL_photo37', 'SNL_photo43', 'SNL_photo44'],
                     'Nacc': ['SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo34', 'SNL_photo35']}

value_change_example_mice = {'tail': 'SNL_photo70', 'Nacc': 'SNL_photo34'}

# state change experiment (block of normal cues and one cue then switched for white noise)
state_change_mice = {'tail': ['SNL_photo70', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo37', 'SNL_photo43'],
                     'Nacc': ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']}
state_change_example_mice = {'tail': 'SNL_photo37', 'Nacc': 'SNL_photo31'}

# Figure 5
bias_analysis_mice = {'tail': [['SNL_photo37', 'SNL_photo22', 'SNL_photo26', 'SNL_photo21'], ['SNL_photo70', 'SNL_photo72', 'SNL_photo57']],
                      'Nacc': ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']}
# supp 6 silence
silence_mice = ['SNL_photo37', 'SNL_photo43', 'SNL_photo70']

# supp 6 movement outside task
out_of_task_mice = ['SNL_photo37', 'SNL_photo43', 'SNL_photo44']
out_of_task_movement_mice_dates = {'SNL_photo37': '20210610_16_55_04', 'SNL_photo43': '20210610_16_20_18', 'SNL_photo44': '20210610_15_45_19'}
