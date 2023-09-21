import os
import pandas as pd
import numpy as np
import seaborn as sns
import sys
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos\\Python_git')
sys.path.insert(0, 'C:\\Users\\francescag\\Documents\\SourceTree_repos')
from first_three_session_cumsum_ang_vel import get_first_three_sessions_dlc
from utils.regression_plotting_utils import make_box_plot
from utils.plotting_visuals import makes_plots_pretty
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from utils.tracking_analysis.dlc_processing_utils import get_camera_trigger_times, find_nearest_trials
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
import scipy as sp
from scipy.stats import shapiro
import matplotlib

font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

# TODO: sort out imports (belong to repo DLC_postprocessing)
# TODO: clean up code

## load group tail data
mouse_ids = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26'] #['SNL_photo28', 'SNL_photo30']#['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35'] #['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26'] #['SNL_photo28', 'SNL_photo30']
site = 'tail'
save = False
num_sessions = 3

data_to_save, all_data = get_first_three_sessions_dlc(mouse_ids, site, save=False, load_saved=True)
q_data = all_data[all_data['recording site'] == site]
s_data = all_data[all_data['recording site'] == 'shuffled '+ site]


## load group nacc data
mouse_ids = ['SNL_photo28', 'SNL_photo30', 'SNL_photo31', 'SNL_photo32', 'SNL_photo33', 'SNL_photo34', 'SNL_photo35'] #['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26'] #['SNL_photo28', 'SNL_photo30']
site = 'Nacc'
save = False
num_sessions = 3

_, all_nacc_data = get_first_three_sessions_dlc(mouse_ids, site, save=False, load_saved=True)
q_nacc_data = all_nacc_data[all_nacc_data['recording site'] == site]
s_nacc_data = all_nacc_data[all_nacc_data['recording site'] == 'shuffled '+ site]

## load example mouse data
example_mouse = 'SNL_photo26'
example_date = '20200810'
example_trial=50
save_out_folder = 'W:\\photometry_2AC\\tracking_analysis\\' + example_mouse
if not os.path.exists(save_out_folder):
    os.makedirs(save_out_folder)
movement_param_file = os.path.join(save_out_folder, 'contra_APE_tracking{}_{}.pkl'.format(example_mouse, example_date))
if os.path.isfile(movement_param_file):
    quantile_data = pd.read_pickle(movement_param_file)

#fig, axs = plt.subplots(1, 1)
#make_box_plot(q_data, dx='recording site', dy='r squared', fig_ax=axs)

### 1. Plot trajectory on photo

def plot_one_trial(x, y, ax=False, cmap='winter', alpha=0.5):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    MAP = cmap
    NPOINTS = x.shape[0]
    cm = plt.get_cmap(MAP)
    ax.set_prop_cycle(cycler('color', [cm(1. * i / (NPOINTS - 1)) for i in range(NPOINTS - 1)]))

    for i in range(NPOINTS - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], alpha=alpha, lw=3)
    return ax

## Load an example trajectory
x, y = (quantile_data['head x'].iloc[example_trial], quantile_data['head y'].iloc[example_trial])

## Load an image
# first, we need to extract a frame number corresponding to the start of the trajectory for the given trial
saving_folder = 'W:\\photometry_2AC\\processed_data\\' + example_mouse + '\\'
restructured_data_filename = example_mouse + '_' + example_date + '_' + 'restructured_data.pkl'
trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
camera_triggers, trial_start_stamps = get_camera_trigger_times(example_mouse, example_date, 'Two_Alternative_Choice')

first_cot = (trial_data[(trial_data['State name'] == 'CueDelay') & (
        trial_data['Instance in state'] == trial_data['Max times in state'])]['Time start'].values * 10000)
first_cot = first_cot.astype(int)

first_cot_triggers = find_nearest_trials(first_cot, camera_triggers)
frame_number = first_cot_triggers[int(quantile_data['trial numbers'].iloc[example_trial])]

infile = "W:\\photometry_2AC\\freely_moving_photometry_data\\SNL_photo26\\20200812_14_49_20\\camera.avi"
# now we load in the frame
frame_dir = os.path.join("C:\\Users\\francescag", "Documents", "exampleVideo")
im = plt.imread(os.path.join(frame_dir, "frame-{}.jpg".format(frame_number)), format="jpeg")

font = {'size': 8}
matplotlib.rc('font', **font)

# overlay
fig, axs = plt.subplots(3, 4, figsize=(8, 5.5))

ax = axs[0, 0]
ax.pcolor(im.mean(axis=2), cmap='Greys_r', rasterized=True)
ax.set_xlim(50, 540)
ax.set_ylim(100, 400)
plot_one_trial(x, y, ax=ax)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax_titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', ' ', 'H', 'I', 'J', 'K']
for i, ax in enumerate(axs.ravel()):
    ax.set_title(ax_titles[i], loc='left', fontweight='bold')

### 2. PLOT HEAD ANGLE
def _plot_one_trial_head_angle(angles, x, y, ax=False, cmap=matplotlib.cm.winter, head_size=50):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    NPOINTS = angles.shape[0]
    colours = cmap(np.linspace(0, 1, NPOINTS))
    for i in range(NPOINTS-1):
        triangle = Polygon(((x[i], y[i]), (x[i] - head_size/2, y[i] + head_size), (x[i] + head_size/2, y[i] + head_size)), fc=colours[i], alpha=0.6)
        transform = Affine2D().rotate_deg_around(*(x[i], y[i]), -angles[i]) + ax.transData
        triangle.set_transform(transform)
        ax.scatter(x[i],y[i], marker='o', s=8, zorder =3, color=colours[i])
        ax.add_artist(triangle)
        #ax.plot([x[i], x[i] + 30*math.cos(angles[i]), x[i] + 30*math.cos(angles[i]), x[i]], [y[i], y[i] + 30*math.sin(angles[i]), y[i] - 30*math.sin(angles[i]), y[i]], alpha=1, lw=3, color=colours[i])
    ax.set_xlim(50, 540)
    ax.set_ylim(100, 400)
    return ax


# make the plot
angles = quantile_data['head angles'].iloc[example_trial][::2]
ax = axs[0, 1]
ax.pcolor(im.mean(axis=2), cmap='Greys_r', rasterized=True)
_plot_one_trial_head_angle(angles, x[::2],y[::2], ax=ax)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

### 3. PLOT EXAMPLE CUMSUM HEAD ANG VEL
ax = axs[0, 2]
cumsum_ang_vel = quantile_data['cumsum ang vel'].iloc[example_trial]
time = np.arange(0, len(cumsum_ang_vel))/30
plot_one_trial(time, cumsum_ang_vel, ax=ax, alpha=1)
ax.set_xlabel('time (s)')
ax.set_ylabel('head angle (degrees)')



##PLOT EXAMPLE SIGMOID FIT
ax = axs[0, 3]
sigmoid_fit = quantile_data['sig y'].iloc[example_trial]

plot_one_trial(time, cumsum_ang_vel, ax=ax, alpha=0.3)
plot_one_trial(time, sigmoid_fit, ax=ax, alpha=1)
max_cumsum_ang_vel = np.max(sigmoid_fit)
ax.axhline(y=max_cumsum_ang_vel, color='k')
ax.set_xlabel('time (s)')
ax.set_ylabel('head angle (degrees)')

##SECOND FIGURE
### 4. PLOT PHOTOMETRY SIGNAL BY QUANTILES


def plot_quantiles_formatted_data(formatted_data, key, sort_by='APE peaks', ax=None, filter_by=None, filter_value=None,
                                  num_divisions=4, colourmap=matplotlib.cm.viridis, plot_means=True, align_end=True):
    colours = colourmap(np.linspace(0, 0.8, num_divisions))
    if ax is None:
        fig, ax = plt.subplots()
    formatted_data = formatted_data.sort_values(by=sort_by, ascending=True, ignore_index=True)
    formatted_data['quantile'] = pd.qcut(formatted_data[sort_by], q=num_divisions)
    num_divisions = formatted_data['quantile'].unique().shape[0]
    all_xs = []
    all_ys = []
    for q in range(0, num_divisions):
        x_s = []
        y_s = []
        lengths = []
        peaks = []
        quantile = formatted_data['quantile'].unique()[q]
        quantile_data = formatted_data.loc[formatted_data['quantile'] == formatted_data['quantile'].unique()[q]]
        if filter_by:
            filtered_data = quantile_data.loc[quantile_data[filter_by] == filter_value]

        else:
            filtered_data = quantile_data
        print(filtered_data.shape)
        for i, row in filtered_data.iterrows():
            if type(key) == str:
                x = row[key]
                if not plot_means:
                    ax.plot(x, color=colours[q], alpha=0.5)
                if not np.isnan(x[0]):
                    x_s.append(x)
                    lengths.append(x.shape[0])
                    peaks.append(row['APE peaks'])
                num_trials = len(x_s)
            else:
                y = row[key[1]]
                x = row[key[0]]
                if not plot_means:
                    ax.plot(x, y, color=colours[q], alpha=0.5)
                if not np.isnan(x[0]):
                    x_s.append(x)
                    y_s.append(y)
                    lengths.append(x.shape[0])
                    peaks.append(row['APE peaks'])
                num_trials = len(x_s)

        if type(key) == str:
            x_array = np.empty((num_trials, max(lengths)))
            x_array[:] = np.nan
            for i in range(0, len(x_s)):
                x_array[i, max(lengths) - lengths[i]:] = x_s[i]
                mean_xs = np.mean(x_array, axis=0)
                mean_xs = mean_xs[np.logical_not(np.isnan(mean_xs))]
                all_xs.append(mean_xs)

            if plot_means:
                if key == "traces":
                    time_points = (np.arange(len(mean_xs)) / 10000) - (len(mean_xs) / 10000 / 2)
                    ax.plot(time_points, mean_xs, color=colours[q], lw=1)
                else:
                    if align_end:
                        time_points = (np.flip(np.arange(len(mean_xs)) * -1) / 30)
                    else:
                        time_points = (np.arange(len(mean_xs)) / 30)
                    ax.plot(time_points, mean_xs, color=colours[q], lw=1)
        else:
            x_array = np.empty((num_trials, max(lengths)))
            x_array[:] = np.nan
            y_array = np.empty((num_trials, max(lengths)))
            y_array[:] = np.nan
            for i in range(0, len(x_s)):
                print(max(lengths) - lengths[i])
                x_array[i, max(lengths) - lengths[i]:] = x_s[i]
                y_array[i, max(lengths) - lengths[i]:] = y_s[i]
            mean_xs = np.mean(x_array, axis=0)
            mean_ys = np.mean(y_array, axis=0)
            all_xs.append(mean_xs)
            all_ys.append(mean_ys)
            if plot_means:
                ax.plot(mean_xs, mean_ys, color=colours[q], lw=1)
    return all_xs


ax = axs[1, 0]
plot_quantiles_formatted_data(quantile_data, "traces", ax=ax)
ax.set_xlabel('time from leaving centre port (s)')
ax.set_ylabel('z-scored fluorescence')
ax.axvline(x=0, color='k')

# ### 6. PLOT TRAJECTORIES BY QUANTILES
# ax = axs[1, 1]
# ax.pcolor(im.mean(axis=2), cmap='Greys_r', rasterized=True)
# plot_quantiles_formatted_data(quantile_data, ['head x', 'head y'], ax=ax)
# ax.set_xlim(50, 540)
# ax.set_ylim(100, 400)
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

### 5. CUMSUM ANG VEL
ax = axs[1, 1]
plot_quantiles_formatted_data(quantile_data, "sig y", ax=ax)
ax.set_xlabel('time from entering choice port (s)')
ax.set_ylabel('head angle (degrees)')
ax.set_ylim(55, 115)
ax.set_xlim(-0.3, 0)
### 6. APE size against max cumsum angular velocity
ax = axs[1, 2]
colourmap = matplotlib.cm.viridis
colours = colourmap(np.linspace(0, 0.8, 4))


quantile_midpoints = []
mean_turns = []
for i, q in enumerate(quantile_data['APE quantile'].unique()[::-1]):
    quantile_midpoint = q.mid
    trials = quantile_data.loc[quantile_data['APE quantile'] == q]
    quantile_midpoints.append(quantile_midpoint)
    mean_turns.append(np.nanmedian(trials['fitted max cumsum ang vel'].values))
    ax.scatter(np.abs(np.nanmedian(trials['fitted max cumsum ang vel'].values)), quantile_midpoint, color=colours[i])


slope, intercept, r_value, p_value, std_err = sp.stats.linregress(mean_turns, quantile_midpoints)

ax.axline((0, intercept), slope=slope, color="k")
ax.set_xlim([70, 130])
ax.set_ylim([0, 4])
ax.set_xlabel('max turn angle')
ax.set_ylabel('fluorescence \n peak size')

### SHOW SUMMARY DATA
site = 'tail'
p_val_data = sp.stats.ttest_ind(all_data[all_data['recording site'] == site]['fit slope'], all_data[all_data['recording site'] == 'shuffled ' + site]['fit slope'])[1]
print('real data p-val:', p_val_data)
p_vals = []
for shuffle_num in s_data['shuffle number'].unique():
    shuffle_data = s_data[s_data['shuffle number'] == shuffle_num]
    p_vals.append(sp.stats.ttest_ind(s_data['fit slope'], shuffle_data['fit slope'])[1])
print('tail proportion of shuffles with p-val <= actual p val:', np.where(np.array(p_vals) <= p_val_data)[0].shape[0]/len(p_vals))

site = 'Nacc'
nacc_p_val_data = sp.stats.ttest_ind(all_nacc_data[all_nacc_data['recording site'] == site]['fit slope'], all_nacc_data[all_nacc_data['recording site'] == 'shuffled ' + site]['fit slope'])[1]
print('real data p-val:', nacc_p_val_data)
nacc_p_vals = []
for shuffle_num in s_nacc_data['shuffle number'].unique():
    shuffle_data = s_nacc_data[s_nacc_data['shuffle number'] == shuffle_num]
    nacc_p_vals.append(sp.stats.ttest_ind(s_nacc_data['fit slope'], shuffle_data['fit slope'])[1])
print('Nacc proportion of shuffles with p-val <= actual p val:', np.where(np.array(nacc_p_vals) <= nacc_p_val_data)[0].shape[0]/len(nacc_p_vals))

all_sites_data = pd.concat([all_data, all_nacc_data])

#fig, axs = plt.subplots(1,3, figsize=(8, 4))
shuffle_compare_boxplot_ax = axs[2, 0]
perc_shuffles_tail_ax = axs[2, 1]
perc_shuffles_nacc_ax = axs[2, 2]
var_exp_ax = axs[2, 3]

custom_set2 = sns.color_palette(['#FDC5AF', '#fc8d62', '#B6E2D4', '#66c2a5',])
make_box_plot(all_sites_data, dx='recording site', dy='fit slope', fig_ax=shuffle_compare_boxplot_ax, scatter_size=3, pal=custom_set2)
shuffle_compare_boxplot_ax.set_xticklabels(['shuffled \n AudS', 'AudS', 'shuffled \n VS', 'VS'], rotation=30)
y = 0.5
h = .1 * y
shuffle_compare_boxplot_ax.plot([0, 0, 1, 1], [y, y+h, y+h, y], c='k', lw=1)
shuffle_compare_boxplot_ax.text(.5, y+h, '***', ha='center', fontsize=12)
shuffle_compare_boxplot_ax.plot([2, 2, 3, 3], [y, y+h, y+h, y], c='k', lw=1)
shuffle_compare_boxplot_ax.text(2.5, y + 2 * h, 'n.s.', ha='center', fontsize=8)
#.set_ylim([None, y + 10 * h])

perc_shuffles_tail_ax.hist(p_vals, color='grey')
perc_shuffles_tail_ax.axvline(p_val_data, color='#fc8d62')
perc_shuffles_tail_ax.set_xlabel('p-val for shuffles')
perc_shuffles_tail_ax.set_ylabel('percentage of shuffles')

#perc_shuffles_nacc_ax.hist(nacc_p_vals, color='grey')
#perc_shuffles_nacc_ax.axvline(nacc_p_val_data, color='#66c2a5')
#perc_shuffles_nacc_ax.set_xlabel('p-val for shuffles')
#perc_shuffles_nacc_ax.set_ylabel('percentage of shuffles')

sns.regplot(data=quantile_data, x='fitted max cumsum ang vel', y='APE peaks', scatter_kws={'s':5, 'color':'#fc8d62'}, ax=perc_shuffles_nacc_ax, color='k', line_kws={'linewidth':1})
example_data = q_data[(q_data['mouse'] == example_mouse) & (q_data['session'] == example_date)]
example_r_squared = example_data['r squared'].values[0]
perc_shuffles_nacc_ax.text(0.9, 0.9, f'R\N{SUPERSCRIPT TWO} = {example_r_squared:.2f}'
                           , horizontalalignment='center', verticalalignment='bottom', transform=perc_shuffles_nacc_ax.transAxes, size=7)
make_box_plot(q_data, dx='recording site', dy='r squared', fig_ax=var_exp_ax, pal=sns.color_palette(['#fc8d62']))
var_exp_ax.set_xlim([-0.5, 0.5])
var_exp_ax.set_xticklabels(['AudS'])


makes_plots_pretty(fig.axes)
plt.tight_layout()

data_directory = 'W:\\thesis\\figures\\'
#plt.savefig(data_directory + 'movement_figure_2.pdf', transparent=True, bbox_inches='tight')
plt.show()
