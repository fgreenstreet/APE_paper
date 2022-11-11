from utils.large_reward_omission_utils import make_example_traces_plot, get_unexpected_reward_change_data_for_site
from utils.plotting_visuals import set_plotting_defaults
sites = ['Nacc', 'tail']

set_plotting_defaults()
for site in sites:
    site_data = get_unexpected_reward_change_data_for_site(site)
    make_example_traces_plot(site, site_data)


