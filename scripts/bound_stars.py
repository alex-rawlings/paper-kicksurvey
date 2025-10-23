import argparse
import os.path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import pygad
from data_classes import RecoilCluster, RecoilClusterSeries
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot bound stellar mass",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
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

# XXX: set the data files and constant quantities we'll need
core_dispersion = 270  # km/s
eff_radius = 5.65  # kpc

if args.minor:
    # apocentre data
    apo_data_files = bgs.utils.get_files_in_dir(
        os.path.join(figure_config.reduced_data_dir_minor, "lagrangian_files/data"),
        ext=".txt",
    )
    # snapshot data
    snapshot_dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/minor_merger/children"
    # main data file
    main_data_dir = os.path.join(figure_config.reduced_data_dir_minor, "bound_stars")
else:
    # apocentre data
    apo_data_files = bgs.utils.get_files_in_dir(
        figure_config.data_path("lagrangian_files/data"),
        ext=".txt",
    )
    # snapshot data
    snapshot_dir = (
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"
    )
    # main data file
    main_data_dir = figure_config.data_path("bound_stars")

if args.extract:
    snap_offset = 3
    for i, apo_file in enumerate(apo_data_files):
        file_name_only = os.path.basename(apo_file).replace(".txt", "")
        _r = np.loadtxt(apo_file, skiprows=1)[snap_offset:, 1]
        if np.any(np.diff(_r) < 0):
            # we have an instance where the distance of the BH to
            # centre is decreasing
            apo_snap_num = np.nanargmax(_r) + snap_offset
        else:
            # no apocentre
            continue
        if np.any(np.diff(_r[apo_snap_num:]) > 0):
            # we have an instance where the distance of the BH to
            # centre is increasing
            peri_snap_num = np.nanargmin(_r[apo_snap_num:]) + apo_snap_num
        else:
            # we have run out of snapshots before pericentre
            SL.warning("No pericentre found, using first 10% of snapshots!")
            peri_snap_num = int(np.ceil(0.1 * len(_r)))
        if "0000" in file_name_only:
            # we have the special 0km/s case
            peri_snap_num = 10
            apo_snap_num = 5

        start_kick_time = datetime.now()
        SL.info(f"Doing kick velocity {file_name_only.replace('kick-vel-','')}")
        snapfiles = [
            bgs.utils.get_snapshots_in_dir(
                os.path.join(snapshot_dir, file_name_only, "output")
            )[sn]
            for sn in [apo_snap_num - 1, apo_snap_num, apo_snap_num + 1, peri_snap_num]
        ]
        clusters = [RecoilCluster() for _ in range(len(snapfiles))]
        for j, snapfile in tqdm(
            enumerate(snapfiles),
            desc="Analysing snapshots",
            total=len(snapfiles),
            disable=True,
        ):
            snap = pygad.Snapshot(snapfile, physical=True)
            if len(snap.bh) != 1:
                raise RuntimeError("We require 1 BH! Skipping this snapshot")
            clusters[j].particle_masses["bh"] = snap.bh["mass"][0]
            clusters[j].particle_masses["stars"] = snap.stars["mass"][0]
            clusters[j].kick_vel = float(
                os.path.splitext(file_name_only)[0].replace("kick-vel-", "")
            )
            clusters[j].time = bgs.general.convert_gadget_time(snap, new_unit="Myr")
            clusters[j].snap_num = bgs.general.get_snapshot_number(snapfile)
            bgs.analysis.basic_snapshot_centring(snap)
            clusters[j].bh_rad = snap.bh["r"].flatten()
            try:
                strong_bound_ids, energy, amb_sigma = (
                    bgs.analysis.find_strongly_bound_particles(snap, return_extra=True)
                )
                clusters[j].ambient_vel_disp = amb_sigma
                bound_id_mask = pygad.IDMask(strong_bound_ids)
                clusters[j].intrinsic_properties["bound_mass"] = np.sum(
                    snap.stars[bound_id_mask]["mass"]
                )
                clusters[j].ids = strong_bound_ids
                clusters[j].intrinsic_properties["vel_disp"] = np.linalg.norm(
                    np.nanstd(snap.stars[bound_id_mask]["vel"], axis=0)
                )
                clusters[j].intrinsic_properties["rhalf"] = float(
                    pygad.analysis.half_mass_radius(
                        snap.stars[bound_id_mask], center=snap.bh["pos"].flatten()
                    )
                )
                _rhalf = np.full(3, np.nan)
                _LOS_sig = np.full(3, np.nan)
                for proj in range(3):
                    _rhalf[proj] = float(
                        pygad.analysis.half_mass_radius(
                            snap.stars[bound_id_mask],
                            proj=proj,
                            center=snap.bh["pos"].flatten(),
                        )
                    )
                    cyl_mask = bgs.analysis.get_cylindrical_mask(
                        _rhalf[proj], proj=proj, centre=snap.bh["pos"].flatten()
                    )
                    compound_mask = bound_id_mask & cyl_mask
                    _LOS_sig[proj] = np.sqrt(
                        float(
                            bgs.mathematics.smooth_bootstrap(
                                snap.stars[compound_mask]["vel"][:, proj][
                                    :, np.newaxis
                                ],
                                sigma=0,
                                statistic=np.nanvar,
                            )[1]
                        )
                    )
                clusters[j].LOS_properties["rhalf"] = _rhalf
                clusters[j].LOS_properties["vel_disp"] = _LOS_sig
            except AssertionError as err:
                SL.exception(err)
                continue

            # clean memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()

        SL.warning(
            f"Completed extraction for {file_name_only} in {datetime.now()-start_kick_time}"
        )
        bgs.utils.save_data(
            {"data": clusters},
            os.path.join(main_data_dir, f"{file_name_only}-bound.pickle"),
        )
        del clusters

# XXX we don't want to go further for minor merger data
if args.minor:
    raise RuntimeError("Stopping! We will not plot minor merger data")

# load the data files
data_files = bgs.utils.get_files_in_dir(main_data_dir, ".pickle")

# set up the figure
fig, ax = plt.subplots(1, 1)
fig.set_figwidth(1.2 * fig.get_figwidth())
ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{km\,s}^{-1}$")
ax.set_ylabel(r"$M/\mathrm{M}_\odot$")


def data_grabber():
    """
    Generator to get the data to plot

    Yields
    ------
    : RecoilClusterSeries
        plotting data
    """
    for i, df in enumerate(data_files):
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


grab_data = data_grabber()
max_r = np.nanmax(list(c.max_rad for c in grab_data))
SL.debug(f"Maximum radius is {max_r}")
r_col_mapper, sm = bgs.plotting.create_normed_colours(
    1e-3, max_r, cmap="rocket", norm="LogNorm"
)

# XXX plot the strongly bound mass
grab_data = data_grabber()
reff_vel = None
for d in grab_data:
    if d.kick_vel < 1 or d.kick_vel > args.maxvel:
        m_bh = d.clusters[0].particle_masses["bh"]
        continue
    ax.semilogy(
        d.kick_vel,
        d.peri.intrinsic_properties["bound_mass"],
        marker="s",
        ls="",
        mew="0.5",
        mec="k",
        c=r_col_mapper(d.peri.bh_rad),
    )
    ax.semilogy(
        d.kick_vel,
        d.apo.intrinsic_properties["bound_mass"],
        marker="o",
        ls="",
        mew="0.5",
        mec="k",
        c=r_col_mapper(d.apo.bh_rad),
    )
    if d.apo.bh_rad > eff_radius and reff_vel is None:
        reff_vel = d.kick_vel
plt.colorbar(sm, ax=ax, label=r"$r/\mathrm{kpc}$", location="top")

ax.scatter([], [], marker="o", c="gray", lw=0.5, ec="k", label=r"$\mathrm{apocentre}$")
ax.scatter([], [], marker="s", c="gray", lw=0.5, ec="k", label=r"$\mathrm{pericentre}$")

# set dual y axis on second plot
SL.debug(f"BH mass is {m_bh:.2e} Msol")
ax.tick_params(axis="y", which="both", right=False)
axr = ax.secondary_yaxis("right", functions=(lambda x: x / m_bh, lambda x: x * m_bh))
axr.set_ylabel(r"$M/M_\bullet$")

# show core dispersion
xlim = ax.get_xlim()
ax.axvspan(xlim[0], core_dispersion, zorder=1, color="gray", alpha=0.6)
ax.text(
    0.1,
    0.4,
    r"$v_\mathrm{kick}< \sigma_{\star,0}$",
    rotation="vertical",
    transform=ax.transAxes,
    va="center",
    # bbox={"fc": "w", "ec": "none"},
)

# show were r_apo > Re
ax.axvline(reff_vel, c="gray", lw=1, ls=":", zorder=0.2)
ax.text(
    1.1 * reff_vel,
    2.8e6,
    r"$r_\mathrm{apo} > R_\mathrm{e}$",
)
ax.annotate(
    "", (700, 3.2e6), (reff_vel, 3.2e6), arrowprops={"arrowstyle": "-|>", "fc": "k"}
)
ax.set_xlim(xlim)

ax.legend(fontsize="small")

bgs.plotting.savefig(figure_config.fig_path("bound.pdf"), force_ext=True)
plt.close()

# XXX: plots for intrinsic properties
fig, ax = plt.subplots(1, 2, sharex="all")
fig.set_figwidth(2 * fig.get_figwidth())
grab_data = data_grabber()
for d in grab_data:
    if d.kick_vel > args.maxvel:
        continue
    for cn, marker in zip(("peri", "apo"), ("s", "o")):
        c = getattr(d, cn)
        SL.debug(
            (
                c.kick_vel,
                c.intrinsic_properties["rhalf"],
                c.intrinsic_properties["vel_disp"],
            )
        )
        ax[0].semilogy(
            c.kick_vel,
            c.intrinsic_properties["rhalf"],
            marker=marker,
            ls="",
            mew="0.5",
            mec="k",
            c=r_col_mapper(c.bh_rad),
        )
        ax[1].semilogy(
            c.kick_vel,
            c.intrinsic_properties["vel_disp"],
            marker=marker,
            ls="",
            mew="0.5",
            mec="k",
            c=r_col_mapper(c.bh_rad),
        )
for axi in ax:
    axi.set_xlabel(r"$v_\mathrm{kick}/\mathrm{km\,s}^{-1}$")
ax[0].set_ylabel(r"$r_{1/2}/\mathrm{kpc}$")
ax[1].set_ylabel(r"$\sigma_\star/\mathrm{km\,s}^{-1}$")
plt.colorbar(sm, ax=ax.flatten(), label="r/kpc", location="top")
ax[0].scatter(
    [], [], marker="o", c="gray", lw=0.5, ec="k", label=r"$\mathrm{apocentre}$"
)
ax[0].scatter(
    [], [], marker="s", c="gray", lw=0.5, ec="k", label=r"$\mathrm{pericentre}$"
)
ax[0].legend()
bgs.plotting.savefig(
    os.path.join(bgs.FIGDIR, "kicksurvey-study/intrinsic_properties.png"), fig=fig
)
plt.close()

fig, ax = plt.subplots()
grab_data = data_grabber()
for d in grab_data:
    if d.kick_vel > args.maxvel:
        continue
    ax.scatter(
        [d.kick_vel] * len(d), d.ambient_sigma_series, c=r_col_mapper(d.bh_radii)
    )
plt.colorbar(sm, ax=ax, label="r/kpc", location="top")
bgs.plotting.savefig(
    os.path.join(bgs.FIGDIR, "kicksurvey-study/ambient_sigma.png"), fig=fig
)
plt.close()
