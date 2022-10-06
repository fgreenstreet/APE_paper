import numpy as np
import pickle
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import matplotlib
from utils.post_processing_utils import remove_manipulation_days, remove_bad_recordings, remove_exps_after_manipulations_not_including_psychometric, get_all_experimental_records, remove_experiments
from utils.plotting import two_conditions_plot, output_significance_stars_from_pval


def get_mean_contra_peak(session_record):
    mouse_id = session_record['mouse_id']
    date = session_record['date']
    saving_folder = 'W:\\photometry_2AC\\processed_data\\' + mouse_id + '\\'
    aligned_filename = mouse_id + '_' + date + '_' + 'aligned_traces.p'
    save_filename = saving_folder + aligned_filename
    session_data = pickle.load(open(save_filename, "rb"))
    _trial_peaks = session_data.choice_data.contra_data.trial_peaks
    trial_peaks = [p if not isinstance(p, np.ndarray) else np.nan for p in _trial_peaks]
    mean_peak = np.nanmean(trial_peaks)
    return mean_peak


def get_session_with_10000th_trial(mouse, experiments):
    dates = experiments[experiments['mouse_id']==mouse]['date'].unique()
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    if session_starts[-1] >= 10000:
        last_session_idx = np.where(np.asarray(session_starts) >=10000)[0][0]
    else:
        last_session_idx = -1
    last_session_date = dates[last_session_idx]
    return last_session_date


def get_first_and_last_peaks(mouse, records, site='tail'):
    experiments_to_process = records[(records['mouse_id'] == mouse) & (records['recording_site'] == site)]
    sorted_records = experiments_to_process.sort_values('date').reset_index()
    first_recording = sorted_records.iloc[0]
    last_recording = sorted_records.iloc[-1]
    first_session_peak = get_mean_contra_peak(first_recording)
    last_session_peak = get_mean_contra_peak(last_recording)
    return first_session_peak, last_session_peak


def get_first_and_10000th_peaks(mouse, records, site='tail'):
    experiments_to_process = records[(records['mouse_id'] == mouse) & (records['recording_site'] == site)]
    sorted_records = experiments_to_process.sort_values('date').reset_index(drop=True)
    first_recording = sorted_records.iloc[0]
    last_recording_date = get_session_with_10000th_trial(mouse, sorted_records)
    last_recording_ind = sorted_records[sorted_records['date'] == last_recording_date].index.values[0]
    last_recording = sorted_records.iloc[last_recording_ind]
    first_session_peak = get_mean_contra_peak(first_recording)
    last_session_peak = get_mean_contra_peak(last_recording)
    return first_session_peak, last_session_peak


def make_beginning_and_end_comparison_plot(ax, site='tail', colour='grey'):
    if site == 'tail':
        mice = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo37', 'SNL_photo43', 'SNL_photo44']
    elif site == 'Nacc':
        mice = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
    records = get_all_experimental_records()
    first_peaks = []
    last_peaks = []
    for mouse in mice:
        all_experiments = remove_exps_after_manipulations_not_including_psychometric(records, [mouse])
        all_experiments = remove_bad_recordings(all_experiments)
        first, last = get_first_and_10000th_peaks(mouse, all_experiments, site=site)
        first_peaks.append(first)
        last_peaks.append(last)

    data = pd.DataFrame({'mouse': mice, 'first session peak mean': first_peaks, 'last session peak mean': last_peaks})

    first_data = data['first session peak mean']
    last_data = data['last session peak mean']
    stat, pval = stats.ttest_rel(first_data, last_data)

    two_conditions_plot(ax, data.set_index('mouse').T, colour=colour, mean_linewidth=0, show_err_bar=False)

    ax.set_xticks([0, 1], ['First session peak', 'Last session peak'], fontsize=6)
    ax.set_ylabel('Z-scored fluorescence', fontsize=6)

    # significance stars
    y = data.set_index('mouse').T.to_numpy().max() + .2
    h = .1
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], c='k', lw=1)
    significance_stars = output_significance_stars_from_pval(pval)
    ax.text(.5, y + h, significance_stars, ha='center', fontsize=8)
    ax.set_ylim([0.3, 2.6])
    plt.tight_layout()
    print(pval)

