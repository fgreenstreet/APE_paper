import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from scipy import stats
import peakutils
from scipy.signal import decimate
from utils.plotting import calculate_error_bars, multi_conditions_plot, output_significance_stars_from_pval
from utils.post_processing_utils import *
from set_global_params import processed_data_path, state_change_example_mice, fig5_plotting_colours


def make_example_plot(site):
    """
    Makes example plot showing one mouse photometry response to white noise and normal cue
    Args:
        site (str): recording site

    Returns:

    """
    mouse_id = state_change_example_mice[site]
    all_experiments = get_all_experimental_records()
    experiment_to_process = all_experiments[
        (all_experiments['experiment_notes'] == 'state change white noise') & (all_experiments['mouse_id'] == mouse_id)]
    session_data, trial_data = open_experiment(experiment_to_process)

    params = {'state_type_of_interest': 5,
              'outcome': 2,
              'last_outcome': 0,  # NOT USED CURRENTLY
              'no_repeats': 1,
              'last_response': 0,
              'align_to': 'Time start',
              'instance': -1,
              'plot_range': [-6, 6],
              'first_choice_correct': 1,
              'cue': 'None'}
    aligned_data = CustomAlignedData(session_data, params)

    trials_pre_state_change = np.where(aligned_data.contra_data.trial_nums <= 149)[0]
    trials_post_state_change = np.where(aligned_data.contra_data.trial_nums > 149)[0]

    fig, axs1 = plt.subplots(1, 1, figsize=[2.5, 2])
    colours =[fig5_plotting_colours[site][0], fig5_plotting_colours[site][-1]]
    all_time_points = decimate(aligned_data.contra_data.time_points, 10)
    start_plot = int(all_time_points.shape[0] / 2 - 2 * 1000)
    end_plot = int(all_time_points.shape[0] / 2 + 2 * 1000)
    time_points = all_time_points[start_plot: end_plot]
    traces = decimate(aligned_data.contra_data.sorted_traces, 10)[:, start_plot: end_plot]
    pre_mean_trace = decimate(np.mean(aligned_data.contra_data.sorted_traces[trials_pre_state_change, :], axis=0), 10)[
                     start_plot: end_plot]
    post_mean_trace = decimate(np.mean(aligned_data.contra_data.sorted_traces[trials_post_state_change, :], axis=0), 10)[
                      start_plot: end_plot]

    axs1.plot(time_points, pre_mean_trace, label='normal cue', color=colours[0])
    pre_error_bar_lower, pre_error_bar_upper = calculate_error_bars(pre_mean_trace,
                                                                    traces[trials_pre_state_change, :],
                                                                    error_bar_method='sem')

    axs1.fill_between(time_points, pre_error_bar_lower, pre_error_bar_upper, alpha=0.4, linewidth=0, color=colours[0])
    axs1.plot(time_points, post_mean_trace, label='white noise', color=colours[1])
    post_error_bar_lower, post_error_bar_upper = calculate_error_bars(post_mean_trace,
                                                                      traces[trials_post_state_change, :],
                                                                      error_bar_method='sem')
    axs1.fill_between(time_points, post_error_bar_lower, post_error_bar_upper, alpha=0.4, linewidth=0, color=colours[1])
    axs1.set_xlim([-2, 2])
    axs1.axvline([0], color='k')
    axs1.set_ylabel('z-scored fluorescence', fontsize=8)
    axs1.set_xlabel('Time (s)', fontsize=8)

    axs1.spines['right'].set_visible(False)
    axs1.spines['top'].set_visible(False)
    axs1.legend(loc='lower left', bbox_to_anchor=(0.7, 0.8),
                borderaxespad=0, frameon=False, prop={'size': 6})
    plt.tight_layout()


def get_group_data(site, save=False):
    """
    Finds data for state change experiment and finds response sizes per mouse before and after state change
    Args:
        site (str): recording site (Nacc or tail)
        save (bool): save out dataframe to csv?

    Returns:
        df_for_plot (pd.dataframe): peak size per mouse pre and post state change
    """
    processed_data_dir = os.path.join(processed_data_path, 'state_change_data')
    state_change_data_file = os.path.join(processed_data_dir, 'state_change_data_{}_mice.csv'.format(site))
    all_session_change_data = pd.read_pickle(state_change_data_file)

    # find mean traces and downsample
    avg_traces = all_session_change_data.groupby(['mouse', 'trial type'])['traces'].apply(np.mean)
    decimated = [decimate(trace[int(len(trace) / 2):], 10) for trace in avg_traces]
    avg_traces = avg_traces.reset_index()
    avg_traces['decimated'] = pd.Series([_ for _ in decimated])

    first_peak_ids = [peakutils.indexes(i)[0] for i in avg_traces['decimated']]
    avg_traces['peakidx'] = first_peak_ids
    peaks = [trace[idx] for idx, trace in zip(first_peak_ids, avg_traces['decimated'])]
    avg_traces['peak'] = peaks
    avg_traces.set_index(['mouse', 'trial type'])

    pre_traces = all_session_change_data[all_session_change_data['trial type'] == 'pre']['traces']
    post_traces = all_session_change_data[all_session_change_data['trial type'] == 'post']['traces']
    pre_traces = pre_traces.reset_index(drop=True)
    post_traces = post_traces.reset_index(drop=True)
    pre_traces_arr = np.zeros([pre_traces.shape[0], pre_traces[0].shape[0]])
    for trial, trace in enumerate(pre_traces):
        pre_traces_arr[trial, :] = trace
    post_traces_arr = np.zeros([post_traces.shape[0], post_traces[0].shape[0]])
    for trial, trace in enumerate(post_traces):
        post_traces_arr[trial, :] = trace

    df_for_plot = avg_traces.pivot(index='trial type', columns='mouse', values='peak').sort_values('trial type',
                                                                                            ascending=False)
    if save:
        df_for_plot.to_csv(os.path.join(processed_data_dir, '{}_peak_sizes.csv'.format(site)))
    return df_for_plot


def pre_post_state_change_plot(df_for_plot, colour='gray'):
    """
    Plots before nad after state change for each mouse
    Args:
        df_for_plot (pd.dataframe): data for all mice pre and post state change
        colour (str): line colour for plot

    Returns:

    """
    pre_peaks = df_for_plot.T['pre'].values
    post_peaks = df_for_plot.T['post'].values
    stat, pval = stats.ttest_rel(pre_peaks, post_peaks)

    fig, ax = plt.subplots(figsize=[2, 2])
    multi_conditions_plot(ax, df_for_plot, mean_linewidth=0, show_err_bar=False, colour=colour)
    plt.xticks([0, 1], ['pre state\nchange', 'post state\nchange'], fontsize=8)
    plt.ylabel('Z-scored fluorescence', fontsize=8)
    ax.set_xlabel(' ')

    # significance stars
    significance_stars = output_significance_stars_from_pval(pval)
    y = df_for_plot.to_numpy().max() + .2
    h = .1
    plt.plot([0, 0, 1, 1], [y, y + h, y + h, y], c='k', lw=1)
    ax.text(.5, y + h, significance_stars, ha='center', fontsize=10)

    plt.tight_layout()

