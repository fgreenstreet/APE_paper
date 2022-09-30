from utils.post_processing_utils import *
import os

site = 'tail'
processed_data_dir = os.path.join('W:\\photometry_2AC\\processed_data\\state_change_data')
state_change_data_file = os.path.join(processed_data_dir, 'state_change_data_{}_mice.csv'.format(site))

mice = ['SNL_photo70', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo37', 'SNL_photo43'] #['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']#['SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo37', 'SNL_photo43', 'SNL_photo44']#['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
for mouse_num, mouse_id in enumerate(mice):
    state_change_data = {}
    exp_type = 'state change white noise'
    all_experiments = get_all_experimental_records()
    all_experiments = remove_bad_recordings(all_experiments)
    experiment_to_process = all_experiments[(all_experiments['experiment_notes'] == exp_type) & (all_experiments['mouse_id'] == mouse_id)]
    session_data = open_experiment(experiment_to_process)[0]

    params = {'state_type_of_interest': 3, # 3 for nacc, 5 for tail
        'outcome': 2,
        'last_outcome': 0,  # NOT USED CURRENTLY
        'no_repeats' : 0,
        'last_response': 0,
        'align_to' : 'Time start',
        'instance': -1,
        'plot_range': [-6, 6],
        'first_choice_correct': 0,
         'cue': 'None'}
    test = CustomAlignedData(session_data, params)

    state_change_data['trial number'] = test.contra_data.trial_nums
    clean_peaks = []
    for peak in test.contra_data.trial_peaks:
        if type(peak) == np.ndarray:
            clean_peaks.append(np.nan)
        else:
            clean_peaks.append(peak)
    state_change_data['peak size'] = clean_peaks #test.contra_data.trial_peaks
    state_change_dataFrame = pd.DataFrame(state_change_data)
    list_traces = [test.contra_data.sorted_traces[i,:] for i in range(test.contra_data.sorted_traces.shape[0])]
    state_change_dataFrame['traces'] = pd.Series(list_traces, index=state_change_dataFrame.index)

    state_change_dataFrame['trial type'] = np.where(state_change_dataFrame['trial number'] < 150, 'pre', 'post')
    state_change_dataFrame['mouse'] = mouse_id
    if mouse_num > 0:
        all_state_change_data = pd.concat([all_state_change_data, state_change_dataFrame], ignore_index=True)
    else:
        all_state_change_data = state_change_dataFrame

#state_change_data_file2 = os.path.join(processed_data_dir, 'state_change_data_tail_mice_test.csv')

all_state_change_data.to_pickle(state_change_data_file)


