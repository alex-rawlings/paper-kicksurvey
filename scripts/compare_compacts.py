import argparse
import os.path
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import baggins as bgs
from data_classes import RecoilClusterSeries
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Compare to compact objects",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-l", "--lower", help="lower velocity", type=float, dest="minvel", default=270
)
parser.add_argument(
    "-u", "--upper", help="upper velocity", type=float, dest="maxvel", default=1080
)
parser.add_argument(
    "--minor", action="store_true", help="analysis for minor mergers", dest="minor"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)

cols = figure_config.custom_colors_shuffled
fig, ax = plt.subplots(1, 2, sharey="all")
fig.set_figwidth(2 * fig.get_figwidth())
fig.set_figheight(1.2 * fig.get_figheight())
rng = np.random.default_rng(42)
vk_cols = figure_config.VkickColourMap()
data_files = bgs.utils.get_files_in_dir(
    figure_config.data_path("bound_stars"), ".pickle"
)
data_files_minor = bgs.utils.get_files_in_dir(
    os.path.join(figure_config.reduced_data_dir_minor, "bound_stars"), ".pickle"
)


def scatter_kwargs_maker():
    """
    Generate scatter kwargs

    Yields
    ------
    : dict
        plotting kwargs
    """
    for col, char in zip(
        figure_config.custom_colors_shuffled, bgs.plotting.mplChars()[1:]
    ):
        yield {
            "mec": "k",
            "mew": 0.1,
            "ms": 2,
            "c": col,
            "fmt": char,
            "elinewidth": 0.5,
            "capsize": 0,
        }


def scatter_kwargs_from_prev(p):
    """
    Get consistent plotting kwargs when data plotted over multiple axes

    Parameters
    ----------
    p : pyplot.ErrorbarContainer
        output from pyplot.errorbar()

    Returns
    -------
    : dict
        plotting kwargs
    """
    return {
        "mec": "k",
        "mew": 0.1,
        "ms": 2,
        "c": p[0].get_color(),
        "fmt": p[0].get_marker(),
        "elinewidth": 0.5,
        "capsize": 0,
    }


def make_cluster_mean_and_error(q, xy):
    """
    Determine the mean and error of cluster properties in log space

    Parameters
    ----------
    q : array-like
        quantity
    xy : str
        x or y quantity to plot

    Returns
    -------
    : dict
        axis mean and error for plt.errorbar()
    """
    xy = xy.lower()
    assert xy == "x" or xy == "y"
    qmean = np.nanmean(np.log10(q))
    qstd = np.nanstd(np.log10(q))
    return {
        xy: 10**qmean,
        f"{xy}err": np.atleast_2d(
            [10**qmean - 10 ** (qmean - qstd), 10 ** (qmean + qstd) - 10**qmean]
        ).T,
    }


def make_cluster_median_and_error(q, xy):
    """
    Determine the median and error of cluster properties in log space

    Parameters
    ----------
    q : array-like
        quantity
    xy : str
        x or y quantity to plot

    Returns
    -------
    : dict
        axis mean and error for plt.errorbar()
    """
    xy = xy.lower()
    assert xy == "x" or xy == "y"
    qmedian = np.nanmedian(np.log10(q))
    qIQR = np.nanquantile(np.log10(q), [0.25, 0.75])
    return {
        xy: 10**qmedian,
        f"{xy}err": np.atleast_2d(
            [10**qmedian - 10 ** qIQR[0], 10 ** qIQR[1] - 10**qmedian]
        ).T,
    }


def data_grabber(minor=False):
    """
    Generator to get the data to plot

    Parameters
    ----------
    minor : bool, optional
        grab minor merger data, by default False

    Yields
    ------
    : RecoilClusterSeries
        plotting data
    """
    if minor:
        _data_files = data_files_minor
    else:
        _data_files = data_files
    for i, df in enumerate(_data_files):
        clusters = bgs.utils.load_data(df)["data"]
        try:
            diff_ids = list(set(clusters[0].ids).difference(set(clusters[1].ids)))
        except TypeError:
            SL.warning(f"No cluster data in {df}, skipping")
            continue
        SL.debug(
            f"{len(diff_ids)/len(clusters[0].ids):.3f} of particles are different between apo and peri centres"
        )
        yield RecoilClusterSeries(*clusters)


def plot_BRC_point(grabbed_data, minor=False):
    if minor:
        marker = "s"
        label = r"$\mathrm{BRC}\;\mathrm{(minor)}$"
    else:
        marker = "o"
        label = r"$\mathrm{BRC}\;\mathrm{(major)}$"
    plot_kwargs = {"ls": "", "marker": marker, "mew": 0.5, "mec": "k"}
    for d in grabbed_data:
        if d.kick_vel > args.maxvel or d.kick_vel < args.minvel:
            continue
        SL.debug(f"Adding vk={d.kick_vel}")
        SL.debug(f"3D density: {d.max_density_2D:.2e} Msol/pc^3")
        c = d.apo
        for ax0, ax1 in zip((ax[0], axins0), (ax[1], axins1)):
            ax0.plot(
                c.intrinsic_properties["bound_mass"],
                c.effective_radius,
                c=vk_cols.get_colour(c.kick_vel),
                **plot_kwargs,
            )
            ax1.plot(
                d.LOS_velocity_dispersion_near_apo,
                c.effective_radius,
                c=vk_cols.get_colour(c.kick_vel),
                **plot_kwargs,
            )
            if minor:
                break
    ax[0].plot([], [], label=label, c="gray", **plot_kwargs)


cluster_plot_kwargs = {
    "fmt": "o",
    "markersize": 4,
    "elinewidth": 0.5,
    "capsize": 0,
    "mec": "k",
    "mew": 0.1,
}

# XXX: FIGURE 1 - MASS VS RE
carlsten20 = bgs.literature.LiteratureTables.load_carlsten_2020_data()
misgeld09 = bgs.literature.LiteratureTables.load_misgeld_2009_data()
misgeld11 = bgs.literature.LiteratureTables.load_misgeld_2011_data()
price09 = bgs.literature.LiteratureTables.load_price_2009_data()
mcconnachie12 = bgs.literature.LiteratureTables.load_mcconnachie_2012_data()
siljeg24 = bgs.literature.LiteratureTables.load_siljeg_2024_data()

sk_gen = scatter_kwargs_maker()

# inset to left panel
axins0 = ax[0].inset_axes(
    [0.6, 0.05, 0.35, 0.2],
    xlim=(1e6, 1.1e7),
    ylim=(32, 75),
    xticklabels=[],
    yticklabels=[],
)
_, connectors = ax[0].indicate_inset_zoom(axins0, edgecolor="k", alpha=1, lw=1)
for i, c in enumerate(connectors):
    c.set_visible(i in [0, 3])
    c.set_linewidth(0.5)

# inset to second panel
axins1 = ax[1].inset_axes(
    [0.6, 0.65, 0.35, 0.3],
    xlim=(700, 1200),
    ylim=(30, 70),
    xticklabels=[],
    yticklabels=[],
)
_, connectors = ax[1].indicate_inset_zoom(axins1, edgecolor="k", alpha=1, lw=1)
for i, c in enumerate(connectors):
    c.set_visible(i in [0, 3])
    c.set_linewidth(0.5)

for axins in (axins0, axins1):
    axins.set_xticks([])
    axins.set_yticks([])

# plot BRC points
grab_data = data_grabber()
plot_BRC_point(grab_data)
if args.minor:
    grab_data_minor = data_grabber(minor=True)
    plot_BRC_point(grab_data_minor, minor=True)


# XXX: add observations
misgeld09.scatter(
    "mass",
    "Re_pc",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="Re_err_pc",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
price09.scatter(
    "mass",
    "Re",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="Re_err",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
# do this data in two parts, as there are a large number of points below Re=10pc
mask = misgeld11.table.loc[:, "Re_pc"] > 10
misgeld11.scatter(
    "mass",
    "Re_pc",
    scatter_kwargs={"fmt": ".", "ms": 1.5, "c": "gray", "zorder": 0.1, "mew": 0},
    ax=ax[0],
    mask=mask,
)
downsampled = np.logical_and(
    ~mask, bernoulli.rvs(0.2, size=misgeld11.num_obs, random_state=rng)
)
misgeld11.scatter(
    "mass",
    "Re_pc",
    scatter_kwargs={"fmt": ".", "ms": 1.5, "c": "gray", "zorder": 0.1, "mew": 0},
    ax=ax[0],
    mask=downsampled,
    use_label=False,
)
_, m12p = mcconnachie12.scatter(
    "mass",
    "rh",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="rh_err",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
carlsten20.scatter(
    "mass",
    "re_pc",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="re_err_pc",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
_, s24p = siljeg24.scatter(
    "mass",
    "Re_pc",
    xerr="mass_err",
    yerr="Re_pc_err",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)

# label some regions of the plot
region_kwargs = {}
ax[0].text(5e3, 4, r"$\mathrm{GCs}$", **region_kwargs)
ax[0].text(3e7, 15, r"$\mathrm{UCDs}$", **region_kwargs)
ax[0].text(4e4, 11, r"$\mathrm{Clusters}$", **region_kwargs)
ax[0].text(5e4, 700, r"$\mathrm{Dwarfs}$", **region_kwargs)
ax[0].text(2e10, 1e4, r"$\mathrm{Bulges}$", **region_kwargs)

ax[0].set_xlim(1e3, ax[0].get_xlim()[1])
ax[0].set_ylim(1, ax[0].get_ylim()[1])
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"$M_\star/\mathrm{M}_\odot$")
ax[0].set_ylabel(r"$R_\mathrm{e}/\mathrm{pc}$")

# XXX: FIGURE 2 - SIGMA VS RE
harris10 = bgs.literature.LiteratureTables.load_harris_2010_data()

harris10.scatter(
    "sig_v",
    "Re",
    xerr="sig_v_err",
    scatter_kwargs=next(sk_gen),
    ax=ax[1],
    use_label=False,
)
mcconnachie12.scatter(
    "vsig",
    "rh",
    xerr="vsig_err",
    yerr="rh_err",
    scatter_kwargs=scatter_kwargs_from_prev(m12p),
    ax=ax[1],
    use_label=False,
)
siljeg24.scatter(
    "vsig",
    "Re_pc",
    xerr="vsig_err",
    yerr="Re_pc_err",
    scatter_kwargs=scatter_kwargs_from_prev(s24p),
    ax=ax[1],
    use_label=False,
)

# label some regions of the plot
ax[1].text(0.5, 4, r"$\mathrm{GCs}$", **region_kwargs)
ax[1].text(1, 1e3, r"$\mathrm{Dwarfs}$", **region_kwargs)

ax[1].set_xlabel(r"$\sigma_\star/\mathrm{km\,s}^{-1}$")
ax[1].set_ylabel("")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
vk_cols.make_cbar(ax=ax[1])

fig.legend(loc="outside upper center", ncols=4)

bgs.plotting.savefig(figure_config.fig_path("compact.pdf"), force_ext=True)
