import os
import matplotlib as mpl
from seaborn import color_palette, cubehelix_palette
import numpy as np

this_dir = os.path.dirname(os.path.realpath(__file__))
reduced_data_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data"
reduced_data_dir_minor = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/minor_mergers"
figure_dir = os.path.abspath(os.path.join(this_dir, "../figures/"))


# get the complete path for saving a figure
def fig_path(fname):
    p = os.path.join(figure_dir, fname)
    if not os.path.isdir(os.path.dirname(p)):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return os.path.join(figure_dir, fname)


# get the complete path to a saved data file
def data_path(fname):
    return os.path.join(reduced_data_dir, fname)


# make sure were using the same settings
mpl.rcdefaults()
mpl.rc_file(os.path.join(this_dir, "matplotlibrc_publish"))

# set the colour map
# colourmap based off seaborn's "icefire" map with 9 colours
col_list = color_palette("icefire", 9).as_hex()
col_list.pop(4)
col_list[4] = "#5e0303"  # slightly different shade for better contrast
# register the original diverging map
custom_diverging_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom_diverging", col_list
)
# register the default map
col_list[4:] = col_list[4:][::-1]  # reverse second half
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", col_list)

# create a custom Blues map, but having the lowest value more blue than white
custom_Blues = mpl.colors.LinearSegmentedColormap.from_list(
    "custom_Blues", mpl.pyplot.cm.Blues(np.linspace(0.25, 1, 256))
)

mpl.colormaps.register(cmap=custom_cmap)  # can use as cmap='custom'
mpl.colormaps.register(cmap=custom_diverging_cmap)
mpl.colormaps.register(cmap=custom_diverging_cmap.reversed(), name="custom_diverging_r")
mpl.colormaps.register(cmap=custom_Blues)

custom_colors = custom_cmap(np.linspace(0, 1, 8))
custom_colors_shuffled = custom_colors[[1, 4, 0, 5, 2, 6, 3, 7]]

color_cycle = mpl.cycler(color=custom_colors)
color_cycle_shuffled = mpl.cycler(color=custom_colors_shuffled)

marker_cycle = mpl.cycler(marker=["o", "s", "^", "D", "v", "p", "h", "X", "P", "*"])
linestyle_cycle = mpl.cycler(ls=["-", ":", "--", "-."])

mpl.rcParams["axes.prop_cycle"] = linestyle_cycle * color_cycle_shuffled

marker_kwargs = {"edgecolor": "k", "lw": 0.5}


class EccentricityScale(mpl.scale.FuncScale):
    """
    A non-linear scale for bound binary eccentricity plots.
    Approximately linear for values below ~0.6, then increasingly non-linear
    to expand the range between 0.9 and 1.
    """

    name = "eccentricity"

    def __init__(self, axis):
        self.fac = 4.5

        def forward(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                res = 1 - 10 ** (-np.arctanh(x) / self.fac)
                return np.where(
                    x > 1, 1000, res
                )  # make invalid values be mapped outside the plot

        def inverse(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.tanh(-self.fac * np.log10((1 - x)))

        super().__init__(axis, (forward, inverse))

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(
            mpl.ticker.FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 0.999, 1])
        )

        axis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:5g}"))


mpl.scale.register_scale(EccentricityScale)


class VkickColourMap:
    """
    Simple class to ensure consistency in how kick velocity is coloured in plots
    """

    def __init__(self) -> None:
        self._vmin = 270  # core dispersion
        self._vmax = 1080  # max velocity that is detevctable
        self.norm = mpl.colors.Normalize(vmin=self._vmin, vmax=self._vmax)
        self.cmapv = cubehelix_palette(
            n_colors=16,
            start=0.0,
            rot=0.5,
            gamma=1.0,
            hue=1.0,
            light=0.9,
            dark=0.1,
            reverse=True,
            as_cmap=True,
        )
        self.cmapv.set_over(mpl.pyplot.get_cmap("Reds")(0.85))
        self.cmapv.set_under("k")
        self.sm = mpl.pyplot.cm.ScalarMappable(norm=self.norm, cmap=self.cmapv)
        self._max_value = -99

    def get_colour(self, v):
        if v > self._max_value:
            self._max_value = v
        return self.cmapv(self.norm(v))

    def make_cbar(self, ax, **kwargs):
        extend = "both" if self._max_value > self._vmax else "min"
        cbar = mpl.pyplot.colorbar(
            self.sm,
            ax=ax,
            label=r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$",
            extend=extend,
            **kwargs,
        )
        return cbar
