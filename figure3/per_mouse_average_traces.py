import pickle
import pandas as pd
import numpy as np
from utils.post_processing_utils import remove_exps_after_manipulations, get_first_x_sessions
from tqdm import tqdm

from set_global_params import experiment_record_path


def get_all_mice_data(experiments_to_process):
    """Note: this function will just pool all mouse data together.
    """
    exp_numbers = []
    mice = []
    for index, experiment in experiments_to_process.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        saving_folder = 'W:\\photometry_2AC\\processed_data\\for_figure\\' + mouse + '\\'
        save_filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'

        sorted_exps = pd.to_datetime(
            experiments_to_process[experiments_to_process['mouse_id'] == mouse]['date']).sort_values(ignore_index=True)
        date_as_dt = pd.to_datetime(date)
        exp_number = sorted_exps[sorted_exps == date_as_dt].index[0]
        exp_numbers.append(exp_number)
        with open(saving_folder + save_filename, "rb") as f:
            session_data = pickle.load(f)
            print(mouse, date)
            if index == 0:
                ipsi_choice = session_data.choice_data.ipsi_data.mean_trace
                contra_choice = session_data.choice_data.contra_data.mean_trace
                reward = session_data.outcome_data.reward_data.mean_trace
                no_reward = session_data.outcome_data.no_reward_data.mean_trace
                time_stamps = session_data.choice_data.contra_data.time_points
            else:
                ipsi_choice = np.vstack([ipsi_choice, session_data.choice_data.ipsi_data.mean_trace])
                contra_choice = np.vstack([contra_choice, session_data.choice_data.contra_data.mean_trace])
                reward = np.vstack([reward, session_data.outcome_data.reward_data.mean_trace])
                no_reward = np.vstack([no_reward, session_data.outcome_data.no_reward_data.mean_trace])
    return ipsi_choice, contra_choice, reward, no_reward, time_stamps


def get_all_mice_average_data(experiments_to_process, time_range=(-1.5, 1.5)):
    """This version takes the average across session for each mouse and then stacks the mouse averages.
    """

    ipsi_choices = []
    contra_choices = []
    rewards = []
    no_rewards = []
    cues = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = 'W:\\photometry_2AC\\processed_data\\for_figure\\' + mouse + '\\'

        mouse_ipsi_choices = []
        mouse_contra_choices = []
        mouse_reward = []
        mouse_no_reward = []
        mouse_cues = []

        for date in df['date']:
            filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'
            with open(data_dir + filename, 'rb') as f:
                session_data = pickle.load(f)
                time_mask = (session_data.choice_data.contra_data.time_points >= time_range[0]) & (session_data.choice_data.contra_data.time_points <= time_range[-1])
                ipsi_choice = session_data.choice_data.ipsi_data.mean_trace[time_mask]
                contra_choice = session_data.choice_data.contra_data.mean_trace[time_mask]
                reward = session_data.outcome_data.reward_data.mean_trace[time_mask]
                no_reward = session_data.outcome_data.no_reward_data.mean_trace[time_mask]
                # for cues you need to take the mean after combining ipsi and contra
                contra_cues = session_data.cue_data.contra_data.sorted_traces[:, time_mask]
                ipsi_cues = session_data.cue_data.ipsi_data.sorted_traces[:, time_mask]
                all_cues = np.concatenate([ipsi_cues, contra_cues])
                mean_trace_cues = np.mean(all_cues, axis=0)
                time_stamps = session_data.choice_data.contra_data.time_points[time_mask]

                mouse_ipsi_choices.append(ipsi_choice)
                mouse_contra_choices.append(contra_choice)
                mouse_reward.append(reward)
                mouse_no_reward.append(no_reward)
                mouse_cues.append(mean_trace_cues)

        ipsi_choices.append(np.mean(mouse_ipsi_choices, axis=0))
        contra_choices.append(np.mean(mouse_contra_choices, axis=0))
        rewards.append(np.mean(mouse_reward, axis=0))
        no_rewards.append(np.mean(mouse_no_reward, axis=0))
        cues.append(np.mean(mouse_cues, axis=0))

    return np.array(ipsi_choices), np.array(contra_choices), np.array(rewards), np.array(no_rewards), np.array(cues), time_stamps


#mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35']
#site = 'Nacc'

mouse_ids = ['SNL_photo17', 'SNL_photo16', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26', 'SNL_photo57', 'SNL_photo58', 'SNL_photo70', 'SNL_photo72']
site = 'tail'

experiment_record = pd.read_csv(experiment_record_path)
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions(all_experiments_to_process).reset_index(
    drop=True)
ipsi_choice, contra_choice, reward, no_reward, cue, time_stamps = get_all_mice_average_data(experiments_to_process)

dir = 'W:\\photometry_2AC\\processed_data\\for_figure\\'
file_name = 'group_data_avg_across_sessions_' + site +'_new_mice_added_with_cues.npz'
np.savez(dir + file_name, ipsi_choice=ipsi_choice, contra_choice=contra_choice, reward=reward, no_reward=no_reward, time_stamps=time_stamps, cue=cue)

