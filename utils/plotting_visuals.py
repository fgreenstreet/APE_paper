import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


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


