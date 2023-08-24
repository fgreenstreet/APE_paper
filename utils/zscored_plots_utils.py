import pickle
import numpy as np
from utils.individual_trial_analysis_utils import SessionData, ZScoredTraces
import matplotlib.pyplot as plt
from scipy.signal import decimate
import seaborn as sns
from matplotlib.colors import ListedColormap
from utils.plotting import calculate_error_bars
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_data_for_figure(recording_site):
    """
    Loads in data for an example mouse for the heatmaps for the heatmaps
    Args:
        recording_site (str): 'VS' or 'TS'

    Returns:
        example_session_data (SessionData): the object containing aligned traces to various behavioural events
    """
    if recording_site == 'VS':
        example_mouse = 'SNL_photo35'
        example_date = '20201119'
    elif recording_site == 'TS':
        example_mouse = 'SNL_photo26'
        example_date = '20200812'

    saving_folder = 'W:\\photometry_2AC\\processed_data\\for_figure\\' + example_mouse + '\\'
    aligned_filename = example_mouse + '_' + example_date + '_' + 'aligned_traces_for_fig.p'
    save_filename = saving_folder + aligned_filename
    example_session_data = pickle.load(open(save_filename, "rb"))
    return example_session_data


def get_correct_data_for_plot(session_data, plot_type):
    """
    Gets traces aligned to correct behavioural events for plot
    Args:
        session_data (SessionData): Object with photometry aligned to all sorts of behavioural events
        plot_type (str): what is the plot aligned to? (ipsi, contra, rewarded, unrewarded)

    Returns:
        output_data (ZScoredTraces): Photometry data relevant to behavioural event needed for plot
    """
    if plot_type == 'ipsi':
        output_data = session_data.choice_data.ipsi_data, 'event end'
    elif plot_type == 'contra':
        output_data = session_data.choice_data.contra_data, 'event end'
    elif plot_type == 'rewarded':
        output_data = session_data.outcome_data.reward_data, 'event start' #'next trial'
    elif plot_type == 'unrewarded':
        output_data = session_data.outcome_data.no_reward_data, 'event start' #'next trial'
    elif plot_type == 'cue':
        output_data = session_data.cue_data.contra_data, 'event start'
    else:
        raise ValueError('Unknown type of plot specified.')
    return output_data


def get_data_for_recording_site(recording_site, ax):
    """
    Gets example mouse data for heatmaps for a given recording site (TS or VS)
    Args:
        recording_site (str): 'VS' or 'TS'
        ax (dict): dict in format {'type of plot': matplotlib.axes._subplots.AxesSubplot}

    Returns:
        axes (list): all plot types for a recording site
        all_data (list): list of ZScoredTraces objects containing relevant to the plot types in axes
        all_white_dot_points (list): list of lists containing reaction times for heatmap white dots
        all_flip_sort_order (list): list of bools (should fastest or slowest reaction time be top?)
        ymins (list): min heatmap value per plot
        ymaxs (list): max heatmap value per plot
    """
    aligned_session_data = get_data_for_figure(recording_site)
    all_data = []
    all_white_dot_points = []
    all_flip_sort_order = []
    ymins = []
    ymaxs = []
    axes = []
    for ax_type, ax in ax.items():
        data, sort_by = get_correct_data_for_plot(aligned_session_data, ax_type)

        if sort_by == 'event end':
            white_dot_point = data.reaction_times
            flip_sort_order = True
        elif sort_by == 'event start':
            white_dot_point = data.reaction_times
            data.reaction_times = data.reaction_times
            flip_sort_order = False
        else:
            raise ValueError('Unknown method of sorting trials')
        all_data.append(data)
        all_white_dot_points.append(white_dot_point)
        all_flip_sort_order.append(flip_sort_order)
        ymin, ymax = get_min_and_max(data)
        ymins.append(ymin)
        ymaxs.append(ymax)
        axes.append(ax[0])
    return axes, all_data, all_white_dot_points, all_flip_sort_order, ymins, ymaxs


def plot_all_heatmaps_same_scale(axes, all_data, all_white_dot_points, all_flip_sort_order, cb_range, cmap='viridis'):
    """
    Plots the heatmaps for traces aligned to different behavioural events using the same colorbar
    Args:
        axes (list): list of axis label/ axis pairs
        all_data (list): list of ZScoredTraces objects to be plotted
        all_white_dot_points (list): list of arrays with the white dot for plots
        all_flip_sort_order (list): list of bools whether to sort by shortest or longest reaction time
        cb_range (list): list of min and max values for each heatmap
        cmap (ListedColormap): colormap to use for heatmap

    Returns:
        heat_map (AxesImage):
    """
    for ax_num, ax_id in enumerate(axes):
        heat_map = plot_heat_map(ax_id, all_data[ax_num], all_white_dot_points[ax_num], all_flip_sort_order[ax_num], dff_range=cb_range, cmap=cmap)
        ax_id.set_xlim([-1.5, 1.5])
        divider = make_axes_locatable(ax_id)
        cax = divider.append_axes("right", size="5%", pad=0.02)
        cb = plt.colorbar(heat_map, cax=cax)
        cb.ax.set_title('z-score', fontsize=8, pad=0.02)
    return heat_map


def get_min_and_max(data):
    """
    Finds min and max of array
    Args:
        data (np.array): data to analyse

    Returns:
        ymin (float): min of array
        ymax (float): max of array
    """
    ymin = np.min(data.sorted_traces)
    ymax = np.max(data.sorted_traces)
    return ymax, ymin

def plot_average_trace(ax, data, error_bar_method='sem', colour='navy'):
    mean_trace = decimate(data.mean_trace, 10)
    time_points = decimate(data.time_points, 10)
    traces = decimate(data.sorted_traces, 10)
    ax.plot(time_points, mean_trace, lw=1, color=colour)# color='navy')

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                traces,
                                                                error_bar_method=error_bar_method)
        ax.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                            facecolor=colour, linewidth=0)


    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-score')


def get_all_mouse_data_for_site(site, file_ext='_new_mice_added_with_cues.npz'):
    dir = 'T:\\photometry_2AC\\processed_data\\for_figure\\'
    file_name = 'group_data_avg_across_sessions_' + site + file_ext  #'group_data_avg_across_sessions_' + site + '.npz'
    data = np.load(dir + file_name)
    return data


def plot_average_trace_all_mice(move_ax, outcome_ax, site, error_bar_method='sem', cmap=sns.color_palette("Set2"), x_range=[-1.5, 1.5]):
    """
    Plots the average trace across mice for movement (contra, ipsi) and outcome (reward, no reward) aligned data
    Args:
        move_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot ipsi/ contra movement aligned traces
        outcome_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot correct/ incorrect outcome aligned data
        site (str): 'VS' or 'TS'
        error_bar_method (str): sem or ci or None
        cmap (list): colours for the two types of data (ipsi vs contra, correct vs incorrect)
        x_range (list): time windon around behavioural event on x-axis

    Returns:

    """
    all_data = get_all_mouse_data_for_site(site, file_ext='_new_mice_added_with_cues.npz')
    time_stamps = all_data['time_stamps']
    data = dict(all_data)
    del data['time_stamps'], data['cue']
    axs = {'contra_choice': move_ax, 'ipsi_choice': move_ax, 'reward': outcome_ax, 'no_reward': outcome_ax}
    colours = {'contra_choice': cmap[0], 'ipsi_choice': cmap[1], 'reward': cmap[0], 'no_reward': cmap[1]}
    for trace_type, traces in data.items():
        mean_trace = decimate(np.mean(traces, axis=0), 10)
        time_points = decimate(time_stamps, 10)
        traces = decimate(traces, 10)
        mean_trace = mean_trace[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        time_points = time_points[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        axs[trace_type].plot(time_points, mean_trace, lw=1, color=colours[trace_type])# color='navy')

        if error_bar_method is not None:
            error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                    traces,
                                                                    error_bar_method=error_bar_method)
            axs[trace_type].fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                                 facecolor=colours[trace_type], linewidth=0)


        axs[trace_type].axvline(0, color='k', linewidth=0.8)
        axs[trace_type].set_xlabel('Time (s)')
        axs[trace_type].set_ylabel('z-score')
        axs[trace_type].set_xlim([-1.5, 1.5])


def plot_average_trace_all_mice_high_low_cues(ax, site, error_bar_method='sem', cmap=sns.color_palette("Set2"), x_range=[-1.5, 1.5]):
    """
    Plots the average trace across mice for movement (contra, ipsi) and outcome (reward, no reward) aligned data
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes to plot high/low cue aligned traces
        site (str): 'VS' or 'TS'
        error_bar_method (str): sem or ci or None
        cmap (list): colours for the two types of data (ipsi vs contra, correct vs incorrect)
        x_range (list): time windon around behavioural event on x-axis

    Returns:

    """

    all_data = get_all_mouse_data_for_site(site, file_ext='_new_mice_added_high_low_cues.npz')
    time_stamps = all_data['time_stamps']
    data = dict(all_data)
    del data['time_stamps']
    axs = {'high_cues': ax, 'low_cues': ax}
    colours = {'high_cues': cmap[0], 'low_cues': cmap[1]}
    labels = {'high_cues': 'high', 'low_cues': 'low'}
    for trace_type, traces in data.items():
        mean_trace = decimate(np.mean(traces, axis=0), 10)
        time_points = decimate(time_stamps, 10)
        traces = decimate(traces, 10)
        mean_trace = mean_trace[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        time_points = time_points[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        axs[trace_type].plot(time_points, mean_trace, lw=1, color=colours[trace_type], label=labels[trace_type])# color='navy')

        if error_bar_method is not None:
            error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                    traces,
                                                                    error_bar_method=error_bar_method)
            axs[trace_type].fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                                 facecolor=colours[trace_type], linewidth=0)


        axs[trace_type].axvline(0, color='k', linewidth=0.8)
        axs[trace_type].set_xlabel('Time (s)')
        axs[trace_type].set_ylabel('z-score')
        axs[trace_type].set_xlim([-1.5, 1.5])
    ax.legend(frameon=False)


def plot_average_trace_all_mice_cue_move_rew(cue_ax, move_ax, outcome_ax, error_bar_method='sem', cmap=sns.color_palette("Set2"), x_range=[-1.5, 1.5]):
    """
    Plots the average trace across mice for movement (contra, ipsi) and outcome (reward, no reward) aligned data
    Args:
        cue_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot contra cue aligned traces
        move_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot contra movement aligned traces
        outcome_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot correct outcome aligned data
        site (str): 'VS' or 'TS'
        error_bar_method (str): sem or ci or None
        cmap (list): colours for the two types of data (TS vs VS)
        x_range (list): time window around behavioural event on x-axis

    Returns:
    """
    sites = ['tail', 'Nacc']
    for i, site in enumerate(sites):
        all_data = get_all_mouse_data_for_site(site, file_ext='_new_mice_added_with_cues.npz')
        time_stamps = all_data['time_stamps']
        data = dict(all_data)
        del data['time_stamps'], data['ipsi_choice'], data['no_reward']
        axs = {'cue': cue_ax, 'contra_choice': move_ax, 'reward': outcome_ax}
        colours = {'contra_choice': cmap[i], 'cue': cmap[i], 'reward': cmap[i]}
        for trace_type, traces in data.items():
            mean_trace = decimate(np.mean(traces, axis=0), 10)
            time_points = decimate(time_stamps, 10)
            traces = decimate(traces, 10)
            mean_trace = mean_trace[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
                x_range[1] * 1000)]
            time_points = time_points[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
                x_range[1] * 1000)]
            traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
                x_range[1] * 1000)]
            axs[trace_type].plot(time_points, mean_trace, lw=1, color=colours[trace_type])# color='navy')

            if error_bar_method is not None:
                error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                        traces,
                                                                        error_bar_method=error_bar_method)
                axs[trace_type].fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                                     facecolor=colours[trace_type], linewidth=0)


            axs[trace_type].axvline(0, color='k', linewidth=0.8)
            axs[trace_type].set_xlabel('Time (s)')
            axs[trace_type].set_ylabel('z-score')
            axs[trace_type].set_xlim([-1.5, 1.5])






def plot_heat_map(ax, data, white_dot_point, flip_sort_order, dff_range=None, x_range=[-1.5, 1.5], cmap='viridis'):
    """
    Plots a single heatmap on a single axis
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes to plot on
        data (ZScoredTraces): photometry traces to plot in heatmap
        white_dot_point (np.array): timestamps (relative to aligned to event) of white dot -normally represents reaction time
        flip_sort_order (bool): shortest or longest reaction time top?
        dff_range (tuple): min and max for colorbar
        x_range (list): time range for x-axis of heatmap (seconds from behavioural event)
        cmap (ListedColormap): colormap for heatmap

    Returns:
        heat_im (AxesImage): the heatmap
    """
    data.sorted_next_poke[-1] = np.nan
    if flip_sort_order:
        arr1inds = (-1 *white_dot_point).argsort()[::-1]
    else:
        arr1inds = white_dot_point.argsort()[::-1]
    data.reaction_times = data.reaction_times[arr1inds]
    data.outcome_times = data.outcome_times[arr1inds]
    data.sorted_traces = data.sorted_traces[arr1inds]
    data.sorted_next_poke = data.sorted_next_poke[arr1inds]

    traces = data.sorted_traces
    clipped_traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 10000): int(traces.shape[1] / 2) + int(x_range[1] * 10000)]
    heat_im = ax.imshow(clipped_traces, aspect='auto',
                        extent=[x_range[0], x_range[1], clipped_traces.shape[0], 0], cmap=cmap)

    ax.axvline(0, color='w', linewidth=1)

    ax.scatter(data.reaction_times,
               np.arange(data.reaction_times.shape[0]) + 0.5, color='w', s=0.5)
    ax.scatter(data.sorted_next_poke,
               np.arange(data.sorted_next_poke.shape[0]) + 0.5, color='k', s=0.5)
    ax.tick_params(labelsize=8)
    ax.set_ylim([data.sorted_traces.shape[0], 0])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial (sorted)')
    if dff_range:
        vmin = dff_range[0]
        vmax = dff_range[1]
        edge = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        heat_im.set_norm(norm)
    return heat_im



def make_y_lims_same_heat_map(ymins, ymaxs):
    ylim_min = min(ymins)
    ylim_max = max(ymaxs)
    return ylim_min, ylim_max