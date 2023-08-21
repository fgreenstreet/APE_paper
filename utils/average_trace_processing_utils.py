import pickle
from tqdm import tqdm
import numpy as np
from set_global_params import processed_data_path


def get_all_mice_average_data(experiments_to_process, time_range=(-1.5, 1.5)):
    """
    This version takes the average across session for each mouse and then stacks the mouse averages.
    Args:
        experiments_to_process (pd.dataframe): experimental records for all the mice you want to average the traces for
        time_range (tuple): time window seconds before and after event to get traces for

    Returns:
        ipsi_choices (np.array): per mouse mean traces aligned to ipsi choices centre port out
        contra_choices (np.array): per mouse mean traces aligned to contra choices centre port out
        rewards (np.array): per mouse mean traces aligned to reward delivery
        no_rewards (np.array): per mouse mean traces aligned to time of choice when there is no reward
        cues (np.array): per mouse mean traces aligned to cues (ipsi and contra)
        time_stamps (np.array): time stamps that correspond to traces (seconds)
    """

    ipsi_choices = []
    contra_choices = []
    rewards = []
    no_rewards = []
    cues = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = processed_data_path + 'for_figure\\' + mouse + '\\'

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
    ipsi_choices = np.array(ipsi_choices)
    contra_choices = np.array(contra_choices)
    rewards = np.array(rewards)
    no_rewards = np.array(no_rewards)
    cues = np.array(cues)
    return ipsi_choices, contra_choices, rewards, no_rewards, cues, time_stamps


def get_all_mice_average_data_only_contra_cues(experiments_to_process, time_range=(-1.5, 1.5)):
    """
    This version takes the average across session for each mouse and then stacks the mouse averages.
    Args:
        experiments_to_process (pd.dataframe): experimental records for all the mice you want to average the traces for
        time_range (tuple): time window seconds before and after event to get traces for

    Returns:
        ipsi_choices (np.array): per mouse mean traces aligned to ipsi choices centre port out
        contra_choices (np.array): per mouse mean traces aligned to contra choices centre port out
        rewards (np.array): per mouse mean traces aligned to reward delivery
        no_rewards (np.array): per mouse mean traces aligned to time of choice when there is no reward
        contra_cues (np.array): per mouse mean traces aligned to contra cues
        time_stamps (np.array): time stamps that correspond to traces (seconds)
    """

    ipsi_choices = []
    contra_choices = []
    rewards = []
    no_rewards = []
    cues = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = processed_data_path + 'for_figure\\' + mouse + '\\'

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
                time_stamps = session_data.choice_data.contra_data.time_points[time_mask]
                mean_trace_cues = np.mean(contra_cues, axis=0)
                mouse_contra_choices.append(contra_choice)
                mouse_reward.append(reward)
                mouse_no_reward.append(no_reward)
                mouse_cues.append(mean_trace_cues)

        ipsi_choices.append(np.mean(mouse_ipsi_choices, axis=0))
        contra_choices.append(np.mean(mouse_contra_choices, axis=0))
        rewards.append(np.mean(mouse_reward, axis=0))
        no_rewards.append(np.mean(mouse_no_reward, axis=0))
        cues.append(np.mean(mouse_cues, axis=0))
    ipsi_choices = np.array(ipsi_choices)
    contra_choices = np.array(contra_choices)
    rewards = np.array(rewards)
    no_rewards = np.array(no_rewards)
    cues = np.array(cues)
    return ipsi_choices, contra_choices, rewards, no_rewards, cues, time_stamps

