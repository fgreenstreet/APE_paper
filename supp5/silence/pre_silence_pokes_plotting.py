import os
import pandas as pd
from set_global_params import processed_data_path
from utils.box_plot_utils import *
from utils.plotting import multi_conditions_plot
from scipy.stats import ttest_rel
filename = os.path.join(processed_data_path, 'num_pokes_in_punishment.pkl')

pre_silence_poke_count = pd.read_pickle(filename)

font = {'size': 8.5, 'family': 'sans-serif', 'sans-serif': ['Arial']}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42

df_for_plot = pre_silence_poke_count.pivot(index='stimulus', columns='mouse', values='count')

fig, ax = plt.subplots(figsize=[1.5, 2])
multi_conditions_plot(ax, df_for_plot, mean_linewidth=0)
ax.set_ylabel('number of pokes', fontsize=7)
# show significance stars
pval = ttest_rel(df_for_plot.T['tones'], df_for_plot.T['silence']).pvalue
y = df_for_plot.to_numpy().max() + .01 * df_for_plot.to_numpy().max()

significance_stars1 = output_significance_stars_from_pval(pval)
ax.text(.5, y, significance_stars1, ha='center', fontsize=8)
plt.tight_layout()
print('mean: ', np.mean(df_for_plot.T['silence']), ', s.d.: ', np.std(df_for_plot.T['silence']))

plt.show()