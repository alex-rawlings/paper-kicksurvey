import argparse
import os.path
import matplotlib.pyplot as plt
import pygad
import baggins as bgs
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Make LSS and instrument plots",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
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

redshifts = (0.2, 0.6, 1)

# read in snapshot, centre on BH
snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0540/output/snap_009.hdf5"
snap = pygad.Snapshot(snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)
bhr = snap.bh["r"].flatten()
pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)

ifu_data_file = os.path.join(figure_config.reduced_data_dir, "instruments.pickle")
if args.extract:
    instruments = [bgs.analysis.ERIS_IFU(), bgs.analysis.JWST_IFU()]
    ifu_data = {f"{instruments[0].name}": [], f"{instruments[1].name}": []}
    for z in redshifts:
        SL.info(f"Doing redshift {z}")
        for instr in instruments:
            instr.redshift = z
            SL.debug(instr)
            SL.info(f"Doing instrument {instr.name}")
            seeing = {"num": 25, "sigma": instr.resolution_kpc}
            ifu_mask = instr.get_fov_mask(0, 2)
            N_stars = len(snap.stars[ifu_mask])
            SL.debug(f"Making IFU for {N_stars} stars")
            assert N_stars > 10
            voronoi = bgs.analysis.VoronoiKinematics(
                x=snap.stars[ifu_mask]["pos"][:, 0],
                y=snap.stars[ifu_mask]["pos"][:, 2],
                V=snap.stars[ifu_mask]["vel"][:, 1],
                m=snap.stars[ifu_mask]["mass"],
                Npx=instr.number_pixels,
                seeing=seeing,
            )
            voronoi.make_grid(part_per_bin=int(75**2))
            voronoi.binned_LOSV_statistics()
            ifu_data[instr.name].append(voronoi.dump_to_dict())
    bgs.utils.save_data(ifu_data, ifu_data_file, exist_ok=True)
ifu_data = bgs.utils.load_data(ifu_data_file)

fig, ax = plt.subplots(3, 3)
for i in range(2):
    ax[0, i + 1].sharex(ax[0, 0])
    ax[0, i + 1].sharey(ax[0, 0])
fig.set_figwidth(2 * fig.get_figwidth())
fig.set_figheight(2 * fig.get_figheight())

for redshift, axi in zip(redshifts, ax[0, :]):
    SL.info(f"Doing z: {redshift}")
    # set up instrument
    micado = bgs.analysis.MICADO_NFM(z=redshift)
    eris = bgs.analysis.ERIS_NIX_NFM(z=redshift)
    eris.max_extent = micado.slit_length_kpc
    jwst = bgs.analysis.JWST_LSS(z=redshift)
    jwst.max_extent = micado.slit_length_kpc

    axi.text(0.1, 0.9, f"$z={redshift:.1f}$", transform=axi.transAxes)

    for instr in (micado, eris, jwst):
        bin_centres, vel_disp = instr.get_LOS_velocity_dispersion_profile(
            snap, N_per_bin=10000
        )

        axi.plot(bin_centres + bhr, vel_disp, alpha=1, label=instr.label)
    axi.set_xlabel(r"$R/\mathrm{kpc}$")
ax[0, -1].legend(fontsize="small")
ax[0, 0].set_ylabel(r"$\sigma/\mathrm{km\,s}^{-1}$")

# IFU plots
counter = 0
sbw = [0.1, 0.2, 0.25, 0.2, 0.3, 0.35]
for axi, (k, v) in zip(ax[1:, :], ifu_data.items()):
    for axii, vv, z in zip(axi, v, redshifts):
        voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(vv)
        voronoi.plot_kinematic_maps(
            ax=axii,
            moments="2",
            cbar="inset",
            clims={"sigma": [180, 220]},
            cbar_kwargs={"ticks": [190, 210], "ha": "left"},
        )
        axii.set_xticks([])
        axii.set_yticks([])
        _text = axii.text(0.75, 0.9, f"$z={z:.1f}$", transform=axii.transAxes)
        _text.set_bbox({"fc": "w", "alpha": 0.4, "ec": "none"})
        bgs.plotting.draw_sizebar(
            ax=axii,
            length=2,
            units="kpc",
            size_vertical=sbw[counter],
            remove_ticks=False,
            location="lower left",
        )
        counter += 1
    if "JWST" in k:
        axi[0].set_ylabel(r"$\mathrm{JWST}$")
    else:
        axi[0].set_ylabel(r"$\mathrm{ERIS}$")

bgs.plotting.savefig(figure_config.fig_path("long_slit_ifu.pdf"), force_ext=True)
