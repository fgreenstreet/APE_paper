import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels as sm
from scipy.stats import ttest_rel
from scipy import stats
from utils.plotting_visuals import makes_plots_pretty
from utils.psychometric_correlation_utils import *
from set_global_params import processed_data_path
import matplotlib

font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

save_path = os.path.join(processed_data_path, 'psychometric_data')
#tail_df = pd.read_csv(processed_data_dir + 'all_trial_data_tail_contra_ipsi_last_trial_confidence.csv')
tail_df = pd.read_pickle(os.path.join(save_path,"all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces_with_tracking_{}_aligned.pkl".format('tail', 'choice')))
#nacc_df = pd.read_pickle(os.path.join(save_path,"all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces_no_tracking_{}_aligned.pkl".format('Nacc', 'reward'))) #for reward
#nacc_df['recording site'] = 'Nacc'
tail_df['recording site'] = 'tail'

#behavioural_df_nacc = nacc_df.dropna().reset_index(drop=True)[['mouse', 'session', 'fiber side','trial numbers', 'trial type', 'side', 'outcome', 'last trial type', 'last choice', 'last outcome', 'next trial type', 'next choice', 'next outcome', 'norm APE', 'recording site']]
behavioural_df_tail = tail_df[['mouse', 'session', 'fiber side','trial numbers', 'trial type', 'side', 'outcome', 'last trial type', 'last choice', 'last outcome', 'next trial type', 'next choice', 'next outcome', 'norm APE', 'recording site']].dropna().reset_index(drop=True)

#df = pd.concat([behavioural_df_nacc, behavioural_df_tail])
# df['percentage high tones'] = df['trial type'].apply(convert)
# df['contra sensory evidence'] = df.apply(convert_lateral,axis=1)
# df['choice sensory evidence'] = df.apply(convert_ipsi_contra,axis=1)
# df['discriminatability'] = df['contra sensory evidence'].apply(convert_difficulty)
# df['numeric side'] = df['side'].apply(convert_ipsi_contra_numeric)
#
# df['previous percentage high tones'] = df['last trial type'].apply(convert)
# df['previous contra sensory evidence']  =  df.apply(convert_lateral_previous,axis=1)
# df['previous discriminatability'] = df['previous contra sensory evidence'].apply(convert_difficulty)
# df['previous choice sensory evidence'] = df.apply(convert_ipsi_contra_last_trial,axis=1)#df['next contra sensory evidence']  = df['next trial type'].apply(convert)
#
# df['next percentage high tones'] = df['next trial type'].apply(convert)
# df['next contra sensory evidence']  =  df.apply(convert_lateral_next,axis=1)
# df['next discriminatability'] = df['next contra sensory evidence'].apply(convert_difficulty)
# df['next choice sensory evidence'] = df.apply(convert_ipsi_contra_next_trial,axis=1)
# df['next numeric side'] = df['next choice'].apply(convert_ipsi_contra_numeric)
# df['shuffled previous discriminatability'] = df['previous discriminatability'].sample(df.shape[0]).values
# df['shuffled next discriminatability'] = df['next discriminatability'].sample(df.shape[0]).values
#
# df_current_correct = df[df.outcome == 1]
# df_c = df_current_correct.reset_index()
# df_c['DA response size'] = df_c.groupby(['mouse', 'recording site'])['norm APE'].apply(categorise_da_responses, cutoff=.65)
# df_t = df_c[df_c['recording site']=='tail']
# df_n = df_c[df_c['recording site']=='Nacc']
#
# diffs_tail = get_diff_psychometrics_da_test(df_t)
# dn = diffs_tail.dropna()
# dn['NextPerceptualUncertainty'] = dn['next contra sensory evidence'].apply(lambda x:1 - np.around(np.abs(.5 - x) ,decimals=2))
# dn['LogNextPerceptualUncertainty'] = dn['NextPerceptualUncertainty'].apply(lambda x:np.log(x))
# key = 'LogNextPerceptualUncertainty'
# dd = dn[[key, 'mouse', 'highdiff', 'lowdiff']]
# new_df = dd.melt(id_vars=[key, 'mouse'])
#
# new_df['normalized_diff'] = (new_df['value'] - new_df['value'].min()) / (new_df['value'].max() - new_df['value'].min())
# new_df['log_normalized_diff'] = np.log( 1 + new_df['normalized_diff'])
# new_df_for_plot = new_df.rename(columns={'value': 'Next trial contralateral bias', 'LogNextPerceptualUncertainty': 'Next trial perceptual uncertainty (log)', 'variable': 'Current trial\n dopamine response'})
# new_df_for_plot['Current trial dopamine response'] = new_df_for_plot['Current trial\n dopamine response'].replace(['highdiff', 'lowdiff'], ['large','small'])
#
# fig, ax = plt.subplots(figsize=[2,2.5])
#
# font = {'size': 7}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['font.sans-serif'] = 'Arial'
# matplotlib.rcParams['font.family']
#
# sns.lineplot(data=new_df_for_plot, x='Next trial perceptual uncertainty (log)', y='Next trial contralateral bias', hue='Current trial\n dopamine response',ci=68, ax=ax)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.legend(['large','small'], title='Current trial\n dopamine response', loc='upper left', frameon=False)
# plt.tight_layout()
# filepath=os.path.join('W:\\paper', 'tail_psychometric_next_trial_bias.pdf' )
# #fig.savefig(filepath)
# plt.pause(10)