import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config


bgs.plotting.check_backend()


parser = argparse.ArgumentParser(
    description="Plot projected density image for 600km/s case",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-n", "--new", help="run new Stan sampling", action="store_true", dest="new"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbose",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", console_level=args.verbose)

rng = np.random.default_rng(42)

gp_kwargs = dict(
    figname_base="gaussian_processes/",
    premerger_ketjufile="/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output",
    rng=rng,
)
if args.new:
    gp = bgs.analysis.VkickApocentreGP(**gp_kwargs)
    data_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/data"
else:
    stan_output_dir = "/scratch/pjohanss/arawling/collisionless_merger/stan_files/gp-vkick-apo/kick-apo/"
    csv_files = bgs.utils.get_files_in_dir(stan_output_dir, ext=".csv")
    timestamp = np.max(
        [int(os.path.basename(cf).split("-")[-1].split("_")[0]) for cf in csv_files]
    )
    SL.warning(f"Using sampling timestamp {timestamp}")

    gp = bgs.analysis.VkickApocentreGP.load_fit(
        f"{stan_output_dir.rstrip('/')}/gp_analytic-{timestamp}*.csv",
        **gp_kwargs,
    )
    data_dir = None

# XXX this has to be determined prior to calling this routine
core_sig = 270
core_rad = 0.58
upper_vk = 1080  # kicks above this do not have a detectable cluster


# the simulated detectability
def threshold_dist(x):
    y = np.atleast_1d(3.21e-2 * x - 11.5)
    y[y < core_rad] = core_rad
    y[x < core_sig] = 1000
    y[x > upper_vk] = 1000
    return y


analysis_params = bgs.utils.read_parameters(
    "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml"
)

gp.extract_data(d=data_dir, maxvel=1080, minvel=300)

SL.info(f"Number of simulations with usable data: {gp.num_groups}")

if args.verbose == "DEBUG":
    gp.print_obs_summary()
    for v, ra in zip(gp.obs_collapsed["vkick"], gp.obs_collapsed["rapo"]):
        print(f"Kick {v:.1e}: {ra:.1e} kpc")

# initialise the data dictionary
gp.set_stan_data()

analysis_params["stan"]["GP_sample_kwargs"]["output_dir"] = os.path.join(
    bgs.DATADIR, "stan_files/gp-vkick-apo/kick-apo"
)


# run the model
gp.sample_model(sample_kwargs=analysis_params["stan"]["GP_sample_kwargs"])

# get fraction of apocentres above X kpc
frac_above_X = gp.fraction_apo_above_threshold(threshold_dist)
print(f"{frac_above_X*100:.3f}% of sampled apocentres are above the threshold function")
frac_above_X_proj = gp.fraction_apo_above_threshold(threshold_dist, proj=True)
print(
    f"{frac_above_X_proj*100:.3f}% of sampled projected apocentres are above the threshold function"
)

# make the plots
fig, ax = plt.subplots(3, 2)
ax = ax.flatten()
for axi in ax[3:]:
    axi.sharex(ax[0])
fig.set_figwidth(2 * fig.get_figwidth())
fig.set_figheight(2.5 * fig.get_figheight())
hdi_levels = [50, 75, 99]

# XXX plot 1: vkick - r_apo relation from GP
ax[0].set_xlabel(gp.input_qtys_labs[0])
ax[0].set_ylabel(gp.folded_qtys_labs[0])
ax[0].set_ylim(0.1, np.max(gp.sample_generated_quantity("y", state="OOS")))
# add a zoom plot for the ~600 km/s regime
axins = ax[0].inset_axes(
    [0.5, 0.05, 0.45, 0.35],
    xlim=(480, 680),
    ylim=(4, 9.95),
    xticklabels=[],
    yticklabels=[],
)
vk_threshold = np.linspace(core_sig + 1, 1.05 * np.max(gp.stan_data["x2"]), 100)

for i, axi in enumerate((ax[0], axins)):
    axi.set_yscale("log")
    gp.posterior_OOS_plot(
        xmodel="x2",
        ymodel=gp.folded_qtys_posterior[0],
        ax=axi,
        smooth=True,
        save=False,
        levels=hdi_levels,
        show_legend=False,
    )
    ylims = ax[0].get_ylim()

    # plot the detection distance thresholds
    (l_rS,) = axi.plot(
        vk_threshold, threshold_dist(vk_threshold), ls="-.", lw=1, c="k", zorder=2
    )
ax[0].legend(loc="upper left")

ax[0].indicate_inset_zoom(axins, ec="k")
axins.set_xticks([])
axins.set_yticks([], minor=True)
ax[0].text(980, threshold_dist(980), r"$r_\mathrm{d}$", va="top")
ax[0].text(750, 8, f"{(1-frac_above_X)*100:.1f}%", va="bottom")
ax[0].text(700, 30, f"{(frac_above_X)*100:.1f}%", va="bottom")
ax[0].set_ylim(0.1, ylims[1])

# XXX plot 2: marginal distribution of apocentre
fill_kwargs = {"alpha": 1, "ec": "k", "lw": 0.2}
gp.plot_generated_quantity_dist(
    ["y"],
    bounds=[(0, 2e2)],
    state="OOS",
    xlabels=gp.folded_qtys_labs,
    ax=ax[1],
    cumulative=True,
    save=False,
    quantiles=[0.5],
    fill_kwargs=fill_kwargs,
)
ax[1].set_xscale("log")
xlim = ax[1].get_xlim()
# show the core region
ax[1].axvspan(ax[1].get_xlim()[0], core_rad, ec="none", fc="lightgray")
ax[1].text(
    0.12,
    0.5,
    r"$r_\mathrm{apo}< r_\mathrm{b,0}$",
    rotation="vertical",
    va="center",
)
ax[1].set_xlim(*xlim)


# XXX plot 3: distribution of apocentre times
gp.plot_apocentre_time_distribution(
    ax=ax[2], cumulative=True, save=False, quantiles=[0.5], fill_kwargs=fill_kwargs
)

# XXX plot 4: vkick - angle offset relation from GP
ax[3].set_xlabel(gp.input_qtys_labs[0])
ax[3].set_ylabel(r"$\theta_\mathrm{min}$")
gp.plot_angle_to_exceed_threshold(
    threshold_dist,
    levels=hdi_levels,
    ax=ax[3],
    save=False,
    smooth_kwargs={"mode": "nearest", "window_length": 5},
)
ax[3].set_xlim(0.8 * core_sig, np.max(gp.stan_data["x2"]))
ax[3].set_ylim(0, 90)
ax[3].text(800, 60, r"$\mathrm{Detectable}$")
ax[3].text(500, 15, r"$\mathrm{Not\; detectable}$")

# XXX plot 5&6: probability of distribution of observable vkicks
bin_width = 100
bins = np.arange(
    -bin_width / 2,  # bin_width*np.ceil(core_sig/bin_width)-bin_width/2,
    bin_width * np.ceil(np.nanmax(gp.stan_data["x2"]) / bin_width) + bin_width / 2,
    bin_width,
)
cols = bgs.plotting.mplColours()[:2][::-1]
gp.plot_observable_fraction(
    threshold_dist, bins=bins, ax=ax[4], cols=cols, save=False, edgecolor="k", lw=0.2
)
gp.plot_observable_fraction(
    threshold_dist,
    bins=bins,
    ax=ax[5],
    cols=cols,
    save=False,
    edgecolor="k",
    lw=0.2,
    cumulative=True,
)
ax[4].set_xlim(-bin_width / 2, bins[-1])
ax[4].set_xlabel(gp.input_qtys_labs[0])
ax[4].set_ylabel(r"$f(v_\mathrm{kick})$")
ax[5].set_xlabel(gp.input_qtys_labs[0])
ax[5].set_ylabel(r"$F(v<v_\mathrm{kick})$")
ax[5].legend()

# add a hash region to indicate velocities for v < sigcore
for axi, ytext in zip((ax[0], ax[3]), (1, 45)):
    axi.axvspan(axi.get_xlim()[0], core_sig, ec="none", fc="lightgray")
    axi.text(
        100,
        ytext,
        r"$v_\mathrm{kick}< \sigma_{\star,0}$",
        rotation="vertical",
        va="center",
    )

# set dual y axis on second plot
for i, axi in enumerate(ax):
    if i in [1, 2]:
        continue
    axi.tick_params(axis="x", which="both", top=False, labelbottom=True)
    axr = axi.secondary_xaxis("top", functions=(lambda x: x / 1800, lambda x: x * 1800))
    axr.set_xlabel(r"$v_\mathrm{kick}/v_\mathrm{esc}$")

bgs.plotting.savefig(figure_config.fig_path("apocentres.pdf"), force_ext=True)
plt.close()

# auxillary plot, not for paper
fig, ax = plt.subplots()
ax.hist(
    gp.stan_data["x2"],
    bins=np.arange(0, np.max(gp.stan_data["x2"]), 100),
    density=True,
    cumulative=True,
)
ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$")
ax.set_ylabel(r"$\mathrm{CDF}$")
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/cumulative_vkick.png"))
plt.close()
