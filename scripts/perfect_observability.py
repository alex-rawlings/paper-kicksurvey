import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Determine cluster observability assuming perfect conditions",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-kv", "--kick-vel", dest="kv", type=int, help="kick velocity", default=600
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-p", "--plot", help="plot snapshot", type=int, dest="plot", default=None
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

data_file_name = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/perfect_obs/perf_obs_{args.kv:04d}.pickle"
os.makedirs(os.path.dirname(data_file_name), exist_ok=True)

snapdir = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output"
snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)

final_snap = bgs.utils.read_parameters(
    "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml"
)["snap_nums"][f"v{args.kv:04d}"]
if final_snap is None:
    final_snap = len(snapfiles)
else:
    final_snap += 1

data = dict(
    snapnums=[],
    times=[],
    cluster_props=[],
)

if args.extract:
    if os.path.exists(data_file_name):
        raise FileExistsError

    bh_rad_prev = -99
    for snapfile in tqdm(snapfiles, desc="Analysing snapshots"):
        snap = pygad.Snapshot(snapfile, physical=True)
        snapnum = bgs.general.get_snapshot_number(snapfile)
        if len(snap.bh) != 1:
            SL.warning(f"Require 1 BH, skipping snapshot {snapnum}")
            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
            continue

        bgs.analysis.basic_snapshot_centring(snap)
        # no point doing this if the BH is a long way away
        bh_rad = snap.bh["r"].flatten()
        if bh_rad > 50:
            SL.warning(f"BH too far away, stopping at snapshot {snapnum}")
            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
            break
        if bh_rad_prev > bh_rad:
            SL.warning("BH passed apocentre, ending")
            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
            break
        else:
            bh_rad_prev = bh_rad
        try:
            data["cluster_props"].append(
                bgs.analysis.observable_cluster_props_BH(snap, proj=1, vel_clip=None)
            )
        except AssertionError as err:
            SL.warning(f"Skipping snapshot {snapnum}: {err}")
            continue
        data["snapnums"].append(snapnum)
        data["times"].append(bgs.general.convert_gadget_time(snap))

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()
    bgs.utils.save_data(data, data_file_name)
else:
    data = bgs.utils.load_data(data_file_name)

visible_count = 0
for snapnum, props in zip(data["snapnums"], data["cluster_props"]):
    if props["visible"]:
        print(f"Observable cluster in snapshot {snapnum}")
        visible_count += 1
print(f"There are {visible_count} snapshots with a visible cluster")

if args.plot is not None:
    subdat = data["cluster_props"][data["snapnums"].index(f"{args.plot:03d}")]
    SL.info(f"Vel sig is {subdat['cluster_vsig']:.2e}")
    fig, ax = plt.subplots(1, 2)
    ax[0].loglog(subdat["r_centres_gal"], subdat["gal_dens"], label="galaxy")
    ax[0].loglog(subdat["r_centres_cluster"], subdat["cluster_dens"], label="cluster")
    ax[1].loglog(
        subdat["r_centres_gal"] - subdat["r_centres_cluster"][0] + 1e-4,
        subdat["gal_dens"],
        label="galaxy",
    )
    ax[1].loglog(
        subdat["r_centres_cluster"] - subdat["r_centres_cluster"][0] + 1e-4,
        subdat["cluster_dens"],
        label="cluster",
    )
    for axi in ax:
        axi.set_xlabel(r"$R/\mathrm{kpc}$")
        axi.set_ylabel(r"$\Sigma(R)/\mathrm{M}_\odot\,\mathrm{kpc}^{-2}$")
    ax[0].legend()
    figdir = os.path.join(bgs.FIGDIR, "kicksurvey-study/perfect_obs")
    os.makedirs(figdir, exist_ok=True)
    bgs.plotting.savefig(
        os.path.join(figdir, f"perf_{args.kv:04d}_snap_{args.plot:03d}.png")
    )
