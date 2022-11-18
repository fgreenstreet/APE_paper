from utils.plotting_visuals import set_plotting_defaults
from utils.value_change_plot_utils import make_example_plot, get_site_data_all_mice, make_group_data_plot
from set_global_params import fig5_plotting_colours
import matplotlib.pyplot as plt

sites = ['Nacc', 'tail']

set_plotting_defaults()
for site in sites:
    site_data, time_points = get_site_data_all_mice(site)
    make_example_plot(site_data, time_points, site)
    make_group_data_plot(site_data, fig5_plotting_colours[site][0])
plt.show()

