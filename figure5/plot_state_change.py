from utils.plotting_visuals import set_plotting_defaults
from utils.state_change_utils import make_example_plot, get_group_data, pre_post_state_change_plot
from set_global_params import fig5_plotting_colours
import matplotlib.pyplot as plt

sites = ['Nacc', 'tail']

set_plotting_defaults()
for site in sites:
    make_example_plot(site)
    site_data = get_group_data(site)
    pre_post_state_change_plot(site_data, colour=fig5_plotting_colours[site][0])

plt.show()


