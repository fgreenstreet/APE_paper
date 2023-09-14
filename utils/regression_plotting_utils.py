from scipy import stats
from scipy.stats import sem, shapiro
from utils.post_processing_utils import remove_exps_after_manipulations, remove_bad_recordings
from utils.linear_regression_utils import *
import os
import seaborn as sns
from tqdm import tqdm
from set_global_params import experiment_record_path, processed_data_path


def get_regression_data_for_plot(recording_site='tail'):
    experiment_record = pd.read_csv(experiment_record_path, dtype=str)
    experiment_record['date'] = experiment_record['date'].astype(str)

    if recording_site == 'tail':
        mouse_ids = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo57',  'SNL_photo70', 'SNL_photo72'] #'SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26',
    elif recording_site == 'Nacc':
        mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo32', 'SNL_photo33',
                     'SNL_photo34', 'SNL_photo35']

    good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    clean_experiments = remove_bad_recordings(good_experiments)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == recording_site)].reset_index(drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process).reset_index(drop=True)

    ipsi_choice_kernels = []
    contra_choice_kernels = []
    ipsi_cue_kernels = []
    contra_cue_kernels = []
    reward_kernels = []
    no_reward_kernels = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        data_dir = processed_data_path + mouse + '\\'
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        mouse_ipsi_choice_kernel = []
        mouse_contra_choice_kernel = []
        mouse_ipsi_cue_kernel = []
        mouse_contra_cue_kernel = []
        mouse_reward_kernel = []
        mouse_no_reward_kernel = []

        for date in df['date']:
            if recording_site == 'tail':
                filename = mouse + '_' + date + '_' +  'linear_regression_kernels_different_shifts_all_cues_matched_trials.p' # 'linear_regression_kernels_different_shifts_not_cleaned.p' #'linear_regression_kernels_different_shifts.p'
            else:
                filename = mouse + '_' + date + '_' + 'linear_regression_kernels_different_shifts_all_cues_matched_trials.p'

            fiber_side = df[df.date == date]['fiber_side'].iloc[0]
            if fiber_side == 'left':
                ipsi_cue = 'high cues'
                contra_cue = 'low cues'
            else:
                ipsi_cue = 'low cues'
                contra_cue = 'high cues'

            with open(data_dir + filename, 'rb') as f:
                session_kernels = pickle.load(f)
                mouse_ipsi_choice_kernel.append(session_kernels['kernels']['ipsi choices'])
                mouse_contra_choice_kernel.append(session_kernels['kernels']['contra choices'])
                mouse_ipsi_cue_kernel.append(session_kernels['kernels'][ipsi_cue])
                mouse_contra_cue_kernel.append(session_kernels['kernels'][contra_cue])
                mouse_reward_kernel.append(session_kernels['kernels']['rewards'])
                mouse_no_reward_kernel.append(session_kernels['kernels']['no rewards'])

        ipsi_choice_kernels.append(np.mean(mouse_ipsi_choice_kernel, axis=0))
        contra_choice_kernels.append(np.mean(mouse_contra_choice_kernel, axis=0))
        ipsi_cue_kernels.append(np.mean(mouse_ipsi_cue_kernel, axis=0))
        contra_cue_kernels.append(np.mean(mouse_contra_cue_kernel, axis=0))
        reward_kernels.append(np.mean(mouse_reward_kernel, axis=0))
        no_reward_kernels.append(np.mean(mouse_no_reward_kernel, axis=0))

    ipsi_choice_kernels = np.array(ipsi_choice_kernels)
    contra_choice_kernels = np.array(contra_choice_kernels)
    ipsi_cue_kernels = np.array(ipsi_cue_kernels)
    contra_cue_kernels = np.array(contra_cue_kernels)
    reward_kernels = np.array(reward_kernels)
    no_reward_kernels = np.array(no_reward_kernels)

    time_stamps = {}

    time_stamps['ipsi choices'] = session_kernels['shifts']['ipsi choices'] / 10000 * 100
    time_stamps['contra choices'] = session_kernels['shifts']['contra choices']  / 10000 * 100
    time_stamps['ipsi cues'] = session_kernels['shifts']['high cues']  / 10000 * 100
    time_stamps['contra cues'] = session_kernels['shifts']['low cues']  / 10000 * 100
    time_stamps['rewards'] = session_kernels['shifts']['rewards']  / 10000 * 100
    time_stamps['no rewards'] = session_kernels['shifts']['no rewards']  / 10000 * 100
    means, sems = organise_data_means(ipsi_choice_kernels, contra_choice_kernels, ipsi_cue_kernels, contra_cue_kernels, reward_kernels, no_reward_kernels)

    # new part where I calculate significance windows for 0.1s bins
    significant_time_bins = {}
    significant_time_stamps, p_choice = calculate_significance_windows(contra_choice_kernels, ipsi_choice_kernels,
                                                             'ipsi choices', time_stamps)
    significant_time_bins['choice'] = significant_time_stamps

    #ipsi contra cues
    significant_time_stamps, p_cue = calculate_significance_windows(contra_cue_kernels, ipsi_cue_kernels,
                                                             'ipsi cues', time_stamps)
    significant_time_bins['cue'] = significant_time_stamps

    #reward no reward
    significant_time_stamps, p_outcome = calculate_significance_windows(reward_kernels, no_reward_kernels,
                                                             'rewards', time_stamps)
    significant_time_bins['outcome'] = significant_time_stamps
    return time_stamps, means, sems, significant_time_bins


def calculate_significance_windows(kernel1, kernel2, time_stamp_key, time_stamps, bin_size=.1, alpha=.01):
    bin_numbers = np.digitize(time_stamps[time_stamp_key],
                              np.arange(time_stamps[time_stamp_key][0], time_stamps[time_stamp_key][-1], bin_size))
    downsampled_kernel1 = np.array(
        [kernel1[:, bin_numbers == i].mean(axis=1) for i in np.unique(bin_numbers)])
    downsampled_kernel2 = np.array(
        [kernel2[:, bin_numbers == i].mean(axis=1) for i in np.unique(bin_numbers)])
    decimated_timestamps = np.array(
        [time_stamps[time_stamp_key][bin_numbers == i].mean() for i in np.unique(bin_numbers)])
    p_vals = []
    for i in range(0, downsampled_kernel2.shape[0]):
        differences = downsampled_kernel1[i, :] - downsampled_kernel2[i, :]
        print(shapiro(differences))
        _, p = stats.mannwhitneyu(downsampled_kernel1[i, :], downsampled_kernel2[i, :])
        p_vals.append(p)
    significant_time_stamps = decimated_timestamps[np.where(np.array(p_vals) < alpha)[0]]
    return significant_time_stamps, p_vals


def make_example_figure(ax1, ax2):
    axs = [ax1, ax2]
    mice_dates = pd.DataFrame(
        {'mouse': ['SNL_photo17', 'SNL_photo35', ], 'date': ['20200206', '20201119', ], 'site': ['tail', 'Nacc']})
    for ind, mouse_date in mice_dates.iterrows():
        mouse = mouse_date['mouse']
        date = mouse_date['date']
        saving_folder = processed_data_path + mouse + '\\'
        save_filename = saving_folder + mouse + '_' + date + '_'
        save_out_filename = saving_folder + mouse + '_' + date + '_example_all_cues_matched_trials.npz'
        mouse_data = np.load(save_out_filename)
        time_stamps = {}
        time_stamps['rewards_ind'] = np.where(mouse_data['rewards'] == 1)
        time_stamps['high_cues_ind'] = np.where(mouse_data['high_cues'] == 1)
        time_stamps['contra_ind'] = np.where(mouse_data['contra_choices'] == 1)
        time_stamps['low_cues_ind'] = np.where(mouse_data['low_cues'] == 1)
        predicted_trace = mouse_data['choice_pred'] + mouse_data['cue_pred'] + mouse_data['outcome_pred']
        axs[ind].plot(mouse_data['dff'], color='k', label='trace')
        #axs[ind].plot(mouse_data['choice_pred'], label='choice kernel', color='#b7094c')
        #axs[ind].plot(mouse_data['cue_pred'], label='cue kernel', color='#90be6d')
        #axs[ind].plot(mouse_data['outcome_pred'], label='outcome kernel', color='#89c2d9')
        axs[ind].plot(predicted_trace, label='predicted trace', color='gray')
        axs[ind].legend()
        axs[ind].axvline(time_stamps['rewards_ind'][0], color='k', lw=0.8)
        axs[ind].text(time_stamps['rewards_ind'][0], -0.1, 'reward', transform=axs[ind].get_xaxis_transform(), size=6)
        axs[ind].axvline(time_stamps['contra_ind'][0], color='k', lw=0.8)
        axs[ind].text(time_stamps['contra_ind'][0], -0.25, 'contra\nchoice', transform=axs[ind].get_xaxis_transform(), size=6)
        if mouse == 'SNL_photo35':
            axs[ind].axvline(time_stamps['high_cues_ind'][0], color='k', lw=0.8)
            axs[ind].text(time_stamps['high_cues_ind'][0], -0.1, 'cue', transform=axs[ind].get_xaxis_transform(), size=6)
        else:
            axs[ind].axvline(time_stamps['low_cues_ind'][0], color='k', lw=0.8)
            axs[ind].text(time_stamps['low_cues_ind'][0], -0.1, 'cue', transform=axs[ind].get_xaxis_transform(), size=6)
        axs[ind].axis('off')

        axs[ind].legend(loc='lower left', bbox_to_anchor=(0.2, -0.6),
                   borderaxespad=0, frameon=False, prop={'size': 8})

def organise_data_means(ipsi_choice_kernel, contra_choice_kernel, ipsi_cue_kernel, contra_cue_kernel, reward_kernel, no_reward_kernel):
    means = {}
    sems = {}
    means, sems = calculate_mean_and_sem(ipsi_choice_kernel, 'ipsi choices', means, sems)
    means, sems = calculate_mean_and_sem(contra_choice_kernel, 'contra choices', means, sems)
    means, sems = calculate_mean_and_sem(ipsi_cue_kernel, 'ipsi cues', means, sems)
    means, sems = calculate_mean_and_sem(contra_cue_kernel, 'contra cues', means, sems)
    means, sems = calculate_mean_and_sem(reward_kernel, 'rewards', means, sems)
    means, sems = calculate_mean_and_sem(no_reward_kernel, 'no rewards', means, sems)
    return means, sems


def calculate_mean_and_sem(kernel, param_name, plotting_dict_means, plotting_dict_sems):
    plotting_dict_means[param_name] = np.mean(kernel, axis=0)
    plotting_dict_sems[param_name] = sem(kernel, axis=0)
    return plotting_dict_means, plotting_dict_sems


def plot_kernels(axs, param_name, means_dict, sems_dict, time_stamps, colour='#7FB5B5', legend=False):
    param_kernel = means_dict[param_name]

    axs.plot(time_stamps[param_name], param_kernel, label=param_name, color=colour)
    axs.axvline(0, color='k')
    axs.fill_between(time_stamps[param_name], param_kernel - sems_dict[param_name],  param_kernel +
                     sems_dict[param_name], alpha=0.5, facecolor=colour, linewidth=0)
    axs.set_ylabel('regression coeff.')
    axs.set_xlabel('time (s)')
    if legend:
        axs.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.6),
            borderaxespad=0, frameon=False, prop={'size': 8})


def plot_kernels_different_shifts(parameter_names, coefs, all_shifts, shift_window_sizes):
    fig, axs = plt.subplots(nrows=1, ncols=len(parameter_names), sharey=True, figsize=(15,8))
    axs[0].set_ylabel('Regression coefficient')
    for param_num, param_name in enumerate(parameter_names):
        shifts = all_shifts[param_num]
        shift_window_size = shift_window_sizes[param_num]
        starting_ind = int(np.sum(shift_window_sizes[:param_num]))
        param_kernel = coefs[starting_ind: starting_ind + shift_window_size]
        axs[param_num].plot(shifts*100/10000, param_kernel, label=param_name)
        axs[param_num].set_title(param_name)
        axs[param_num].axvline([0])
        axs[param_num].set_xlabel('Time (s)')


def plot_kernels_for_site(move_axs, cue_axs, reward_axs, means, sems, time_stamps, palette=['#E95F32', '#F9C0AF'], legend=False):
    plot_kernels(move_axs, 'ipsi choices', means, sems, time_stamps, colour=palette[1], legend=legend)
    plot_kernels(move_axs, 'contra choices', means, sems, time_stamps, colour=palette[0], legend=legend)
    plot_kernels(cue_axs, 'ipsi cues', means, sems, time_stamps, colour=palette[1], legend=legend)
    plot_kernels(cue_axs, 'contra cues', means, sems, time_stamps, colour=palette[0], legend=legend)
    plot_kernels(reward_axs, 'rewards', means, sems, time_stamps, colour=palette[0], legend=legend)
    plot_kernels(reward_axs, 'no rewards', means, sems, time_stamps, colour=palette[1], legend=legend)


def load_exp_var_data_for_site(site):
    if site == 'tail':
        mice = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo57',
                'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
        file_name = site + '_explained_variances_all_cues.p'
    elif site == 'Nacc':
        mice = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33',
                     'SNL_photo34', 'SNL_photo35']
        file_name = site + '_explained_variances_all_cues.p'
    processed_data_dir = os.path.join('W:\\photometry_2AC\\processed_data\\linear_regression_data\\')
    saving_filename = os.path.join('W:\\photometry_2AC\\processed_data\\linear_regression_data\\', file_name)

    reg_stats = pd.read_pickle(saving_filename)
    reg_stats = reg_stats[reg_stats['mouse_id'].isin(mice)]
    mean_stats = reg_stats.groupby(['mouse_id'])[ ['cue explained variance', 'choice explained variance', 'outcome explained variance', 'full model explained variance']].apply(np.mean)
    types = []
    variances = []
    for ind, row in mean_stats.iterrows():
        types.append('cue')
        types.append('choice')
        types.append('outcome')
        types.append('full')
        variances.append(row['cue explained variance'])
        variances.append(row['choice explained variance'])
        variances.append(row['outcome explained variance'])
        variances.append(row['full model explained variance'])
    stats_dict = {'predictor': types, 'explained variance': variances}
    reshaped_stats = pd.DataFrame(stats_dict)
    if site == 'Nacc':
        label = 'VS'
    elif site == 'tail':
        label = 'AudS'
    reshaped_stats['site'] = label
    return reshaped_stats


def get_data_both_sites_for_predictor(nacc_data, tail_data, predictor):
    df = pd.concat([nacc_data[nacc_data['predictor']==predictor], tail_data[tail_data['predictor']==predictor]])
    return df


def make_box_plot(df, fig_ax,  dx ='site', dy = 'explained variance', ort = "v", pal = ['#E95F32', '#002F3A'], set_ylims=False, label=None, scatter_size=4):
    custom_palette = sns.set_palette(sns.color_palette(pal))
    TS_data = df[df.site == 'AudS']
    TS_noise = np.random.normal(0, 0.04, TS_data.shape[0])
    VS_data = df[df.site == 'VS']
    VS_noise = np.random.normal(0, 0.04, VS_data.shape[0])
    fig_ax.scatter((TS_data[dx].values == 'AudS').astype(int) + TS_noise + 0.3, TS_data[dy].values, color='#002F3A', s=7, alpha=0.6)
    fig_ax.scatter((VS_data[dx].values == 'AudS').astype(int) + VS_noise - 0.3, VS_data[dy].values, color='#E95F32',
                   s=7, alpha=0.6)
    df = df.replace('AudS', 'TS')
    sns.boxplot(x=dx, y=dy, data=df, palette=custom_palette, width = .3, zorder = 10,linewidth=0.1, \
                showcaps = True, boxprops = {"zorder":10},\
                showfliers=False, whiskerprops = {'linewidth':0.5, "zorder":10},\
                   saturation = 1, orient = ort, ax=fig_ax,
                 medianprops={'color':'white', 'linewidth':1})
    fig_ax.set_xlim([-1, 2])
    if set_ylims:
        fig_ax.set_ylim([-2, np.max(df[dy]) + 2])
    if label:
        fig_ax.text(0.5, 1, label, transform=fig_ax.get_xaxis_transform(), size=8, ha='center')