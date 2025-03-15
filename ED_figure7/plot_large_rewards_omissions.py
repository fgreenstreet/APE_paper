from utils.large_reward_omission_utils import make_example_traces_plot, get_unexpected_reward_change_data_for_site, compare_peaks_across_trial_types
from utils.plotting_visuals import set_plotting_defaults
from set_global_params import fig4_plotting_colours
import matplotlib.pyplot as plt

sites = ['Nacc', 'tail']

set_plotting_defaults()
for site in sites:
    site_data = get_unexpected_reward_change_data_for_site(site)
    make_example_traces_plot(site, site_data)
    compare_peaks_across_trial_types(site_data, colour=fig4_plotting_colours[site][0])
plt.show()


