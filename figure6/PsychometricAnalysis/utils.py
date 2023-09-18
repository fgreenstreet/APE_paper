import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import matplotlib


def makes_plots_pretty(axs):
    if type(axs) == np.ndarray or type(axs) == list:
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
    else:
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.yaxis.set_ticks_position('left')
        axs.xaxis.set_ticks_position('bottom')


def set_plotting_defaults(font_size=7, font_style='Arial'):
    font = {'size': font_size}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.sans-serif'] = font_style
    matplotlib.rcParams['font.family']


# Define mapping for trial type to percentage high tones
PERCENTAGE_HIGH_TONES = {
    1.: .98,
    2.: .82,
    3.: .66,
    4.: .50,
    5.: .34,
    6.: .18,
    7.: .02
}

# Function to convert trial type to percentage high tones
def convert_trial_type_to_percentage(trial_type):
    return PERCENTAGE_HIGH_TONES[trial_type]

# General function to determine the sensory evidence based on the fiber side
def get_sensory_evidence(data, column_name):
    if data['fiber side'] == 'right':
        return data[column_name]
    return round(1 - data[column_name], 2)

# Function to convert side to numeric value
def convert_side_to_numeric(side):
    mapping = {'ipsi': 0, 'contra': 1}
    return mapping[side]

# Calculate the discriminatability based on sensory evidence
def calculate_discriminability(sensory_evidence):
    return np.round(-(0.5 - sensory_evidence) / 0.5, 2)


def categorise_da_responses(response, cutoff=.65):
    """Label DA responses as "low" or "high" based on quantile cutoff value. Default cutoff value based on Lak et al.
    """
    return pd.qcut(response, [0., cutoff, 1.], labels=['small','large'])


def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a + b * x)))


def get_diff_in_proportion_correct(df):
    # Group by mouse and contraSensoryEvidence to compute the proportion of correct trials
    outcome_counts = df.groupby(['mouse', 'contraSensoryEvidence', 'outcome']).size().unstack().fillna(0)
    outcome_counts['proportion_correct'] = outcome_counts[1] / (outcome_counts[0] + outcome_counts[1])

    # Reset index for easier processing
    outcome_counts_reset = outcome_counts.reset_index()

    # Get the highest and lowest contraSensoryEvidence levels
    max_evidence = outcome_counts_reset['contraSensoryEvidence'].max()
    min_evidence = outcome_counts_reset['contraSensoryEvidence'].min()

    # Filter out rows corresponding to the highest and lowest levels
    max_rows = outcome_counts_reset[outcome_counts_reset['contraSensoryEvidence'] == max_evidence]
    min_rows = outcome_counts_reset[outcome_counts_reset['contraSensoryEvidence'] == min_evidence]

    # Merge the two filtered DataFrames on the 'mouse' column
    merged = pd.merge(max_rows, min_rows, on='mouse', suffixes=('_max', '_min'))

    # Compute the difference in proportion correct
    merged['difference'] = np.abs(merged['proportion_correct_max'] - merged['proportion_correct_min'])

    # Extract relevant columns
    difference_df = merged[['mouse', 'difference']]

    return difference_df


def output_significance_stars_from_pval(pval):
    if pval >= 0.05:
        return 'n.s.'
    elif (pval < 0.05) & (pval >= 0.01):
        return '*'
    elif (pval < 0.01) & (pval >= 0.001):
        return '**'
    elif pval < 0.001:
        return '***'
def calculate_statistics(df):
    """Calculate some useful statistics for the psychometric analysis.
    """
    # Compute some statistics
    df['percentageHighTones'] = df['trial type'].apply(convert_trial_type_to_percentage)
    df['contraSensoryEvidence'] = df.apply(lambda x: get_sensory_evidence(x, 'percentageHighTones'), axis=1)
    df['choiceSensoryEvidence'] = df.apply(lambda x: get_sensory_evidence(x, 'contraSensoryEvidence'), axis=1)
    df['discriminatability'] = df['contraSensoryEvidence'].apply(calculate_discriminability)
    df['numericSide'] = df['side'].apply(convert_side_to_numeric)
    df['nextPercentageHighTones'] = df['next trial type'].apply(convert_trial_type_to_percentage)
    df['nextContraSensoryEvidence'] = df.apply(lambda x: get_sensory_evidence(x, 'nextPercentageHighTones'), axis=1)
    df['nextDiscriminability'] = df['nextContraSensoryEvidence'].apply(calculate_discriminability)
    df['nextChoiceSensoryEvidence'] = df.apply(lambda x: get_sensory_evidence(x, 'nextContraSensoryEvidence'), axis=1)
    df['nextNumericSide'] = df['next choice'].apply(convert_side_to_numeric)
    df['nextStaySwitch'] = (df['side'] == df['next choice']).astype(int)
    return df


def calculate_psychometric(df):
    """Fit psychometric curves and get some stats."""
    # Placeholder for bias values
    mouse_biases = []
    # For each mouse, fit the logistic function and compute the PSE
    for mouse in df['mouse'].unique():
        mouse_data = df[df['mouse'] == mouse]

        # Fit the logistic function to the mouse's data
        popt, _ = curve_fit(logistic, mouse_data['contraSensoryEvidence'], mouse_data['numericSide'])

        a, b = popt
        pse = -a / b
        slope_at_pse = .25 * b

        mouse_biases.append({'mouse': mouse, 'pse': np.abs(pse - .5), 'slope': slope_at_pse, 'a': a, 'b': b})

    # Convert to DataFrame
    psychometric_df = pd.DataFrame(mouse_biases)

    return psychometric_df
