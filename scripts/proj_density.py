import argparse
import os.path
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import pygad
import re
import h5py
import baggins as bgs
import dask
import figure_config

bgs.plotting.check_backend()

TARGET_PROM_SCALE = 0.25


# class to facilitate different use cases
class ProjectedDensityObject:
    def __init__(self, redshift, logger, plot=False):
        self._redshift = redshift
        self._logger = logger
        self._plot = plot
        self._snapfiles = None
        self.instrument = bgs.analysis.Euclid_VIS(z=redshift)
        self.instrument.max_extent = 40
        self.filter_code = "Euclid/VIS.vis"
        self.galaxy_metallicity = 0.03396487304923489
        self.galaxy_star_age = 3.645e9  # yr
        self.binary_core_radius = 0.58
        self._save_location = None
        self.max_bh_dist = 30
        self._cluster_prom = None

        # plotting attributes
        self.ax = None
        self.xaxis = 0
        self.yaxis = 2
        self._fontsize = 12
        self.prom_scale = None

    @property
    def single_snapshot(self):
        return self._single_snapshot

    @property
    def save_location(self):
        return self._save_location

    @save_location.setter
    def save_location(self, v):
        self._save_location = v

    @property
    def cluster_prom(self):
        return self._cluster_prom

    @classmethod
    def load_single_snapshot(cls, snapfile, redshift, logger):
        C = cls(redshift=redshift, logger=logger)
        C._plot = True
        C._snapfiles = [snapfile]
        C.setup_plot()
        C._single_snapshot = True
        return C

    @classmethod
    def load_snapshot_list(cls, snapdir, redshift, saveloc, logger, snapnums=None):
        C = cls(redshift=redshift, logger=logger)
        snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)
        if snapnums is None:
            C._snapfiles = snapfiles
        else:
            C._snapfiles = [
                s
                for s in snapfiles
                if any(sn in os.path.basename(s) for sn in snapnums)
            ]
        assert os.path.splitext(saveloc)[1] == ".pickle"
        C.save_location = saveloc
        C._single_snapshot = False
        return C

    def setup_plot(self):
        fig, self.ax = plt.subplots(1, 3)
        fig.set_figwidth(3 * fig.get_figwidth())

    def add_extras(self, ax, pos):
        # mark BH position
        annotate_str = r"$\mathrm{BRC}$"
        ax.annotate(
            annotate_str,
            (pos[0, 0] + 0.25, pos[0, 2] - 0.25),
            (pos[0, 0] + 10, pos[0, 2] - 10),
            color="w",
            arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
            ha="right",
            va="bottom",
            fontsize=self._fontsize,
        )
        ax.set_facecolor("k")
        ax.text(
            0.05,
            0.9,
            f"$z={self._redshift:.1f}$",
            color="w",
            transform=ax.transAxes,
            fontsize=self._fontsize,
        )

    def run(self, save_figure=True, dump=False, sigma=None):
        data_to_save = {}
        prev_bh_dist = -99

        # calculations that needn't be done each iteration
        density_mask = pygad.ExprMask("abs(pos[:,0]) <= 25") & pygad.ExprMask(
            "abs(pos[:,2]) <= 25"
        )
        synth_grid, synth_SED = bgs.analysis.get_spectrum_ssp(
            age=self.galaxy_star_age, metallicity=self.galaxy_metallicity
        )
        euclid_filters = bgs.analysis.get_euclid_filter_collection(synth_grid)

        for snapnum, snapfile in enumerate(self._snapfiles):
            # load and centre the snapshot
            snap = self.load_and_centre_snap(snapfile=snapfile)

            if len(snap.bh) > 1 or snap.bh["r"].flatten() < self.binary_core_radius:
                self._logger.warning(
                    "BHs have not yet merged or BH still within core radius! Skipping this snapshot"
                )
                # clean memory
                snap.delete_blocks()
                del snap
                pygad.gc_full_collect()
                continue

            # check to see if we have reached apocentre: if so, break
            current_bh_dist = pygad.utils.geo.dist(snap.bh["pos"][0, :])
            if current_bh_dist < prev_bh_dist or current_bh_dist > self.max_bh_dist:
                self._logger.warning("We have reached apocentre! Stopping")
                bgs.utils.save_data(data_to_save, self.save_location)
                break
            else:
                prev_bh_dist = current_bh_dist

            if self.single_snapshot:
                t0 = None
                _snapfiles = bgs.utils.get_snapshots_in_dir(os.path.dirname(snapfile))
                i = 0
                while t0 is None:
                    self._logger.debug(f"Finding t0: iteration {i}")
                    with h5py.File(_snapfiles[i], "r") as f:
                        if len(f["/PartType5/Masses"]) < 2:
                            _t0snap = pygad.Snapshot(_snapfiles[i], physical=True)
                            t0 = bgs.general.convert_gadget_time(_t0snap)
                            del _t0snap
                    i += 1
                self._logger.info(
                    f"Snapshot is at time {bgs.general.convert_gadget_time(snap)-t0:.3f} Gyr"
                )

            if self._plot:
                self.plot_surface_mass_density(snap=snap, density_mask=density_mask)

            # convert to magnitudes
            bgs.analysis.set_luminosity(snap=snap, sed=synth_SED, z=self._redshift)

            # now we need to bin the galaxy in the 2D plane,
            # and determine the magnitude for each pixel
            start_time = datetime.now()
            star_count, xedges, yedges = np.histogram2d(
                x=snap.stars[density_mask]["pos"][:, self.xaxis],
                y=snap.stars[density_mask]["pos"][:, self.yaxis],
                bins=self.instrument.number_pixels,
            )
            self._logger.debug(f"There are {(len(xedges)-1)**2:.2e} bins")

            # helper function to parallelise magnitude calculation
            @dask.delayed
            def parallel_mag_helper(num_stars_in_bin):
                nonlocal snap
                return bgs.analysis.get_surface_brightness(
                    sed=synth_SED,
                    stellar_mass=num_stars_in_bin * float(snap.stars["mass"][0]),
                    filters_collection=euclid_filters,
                    filter_code=self.filter_code,
                    z=self._redshift,
                    pixel_size=xedges[1] - xedges[0],
                )["app_mag"]

            res = []
            for sc in star_count.flat:
                res.append(parallel_mag_helper(sc))
            res = dask.compute(*res)
            mag_map = np.array(res).reshape(star_count.shape).T
            self._logger.info(
                f"Apparent magnitude calculated in {datetime.now()-start_time}"
            )

            if self._plot:
                ax_mag = self.ax[1]
            else:
                # define a dummy axis
                fig, ax_mag = plt.subplots()
            im_mag = ax_mag.imshow(
                mag_map,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="mako_r",
            )
            if dump:
                # XXX this is strictly for debugging purposes and for testing
                # prominence methods
                _debug_data = dict(
                    mags=im_mag.get_array(),
                    bh=[snap.bh["pos"][0, self.xaxis], snap.bh["pos"][0, self.yaxis]],
                    xedges=xedges,
                    yedges=yedges,
                    filename=snapfile,
                )
                bgs.utils.save_data(
                    _debug_data,
                    "/users/arawling/projects/collisionless-merger-sample/code/misc/recoil-explore/mag_data.pickle",
                )

            if self._plot:
                self.plot_app_magnitude(im_mag=im_mag, snap=snap)
            else:
                plt.close()
                k = f"snap_{snapnum:03d}"
                data_to_save[k] = {}
                data_to_save[k]["snap"] = snapfile
                data_to_save[k]["im_mag"] = im_mag

            # determine a local "prominence"
            self.calculate_prominence(
                im_mag, snap, save_figure=save_figure, sigma=sigma
            )

            # clean memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
        else:
            if not self._plot:
                bgs.utils.save_data(data_to_save, self.save_location)

    def load_and_centre_snap(self, snapfile):
        self._logger.info(f"Reading snapshot {snapfile}")
        snap = pygad.Snapshot(snapfile, physical=True)
        # move to CoM frame
        bgs.analysis.basic_snapshot_centring(snap)
        return snap

    def calculate_prominence(
        self, im_mag, snap, sigma=None, detection=4.0, save_figure=True
    ):
        if isinstance(snap, str):
            snap = self.load_and_centre_snap(snap)

        log_image = np.clip(np.log10(im_mag.get_array()), 1e-6, None)

        if sigma is None:
            self._logger.debug(f"Target prominence scale: {TARGET_PROM_SCALE}")
            sigma = TARGET_PROM_SCALE / self.instrument.pixel_width
        self._logger.warning(f"Using a prominence sigma of {sigma:.2e}")
        # convolve with a low-pass filter and standardise
        prom = log_image - gaussian_filter(log_image, sigma, mode="nearest")
        prom = -(prom - np.mean(prom)) / np.std(prom)
        # set the edges where the kernel extends beyond the image to 0
        edge_idx = int(np.ceil(sigma * 4) - 1)  # truncated to 4 sigma
        prom[:edge_idx, :] = 0
        prom[-edge_idx:, :] = 0
        prom[:, :edge_idx] = 0
        prom[:, -edge_idx:] = 0

        if self._plot:
            ax_prom = self.ax[2]
        else:
            # define a dummy axis
            fig, ax_prom = plt.subplots()
        im_SN = ax_prom.imshow(
            prom,
            origin="lower",
            extent=im_mag.get_extent(),
            cmap="RdBu_r",
            norm=CenteredNorm(0, halfrange=self.prom_scale),
        )
        if self._plot:
            self.plot_prominence_map(im_SN=im_SN, snap=snap)
            if save_figure:
                bgs.plotting.savefig(
                    figure_config.fig_path("density_map.pdf"),
                    fig=self.ax[0].get_figure(),
                    force_ext=True,
                )
            plt.close()

        signal_prom = bgs.mathematics.get_pixel_value_in_image(
            snap.bh["pos"][0, self.xaxis], snap.bh["pos"][0, self.yaxis], im_SN
        )[0]
        self._logger.info(f"Prominence at BH position is {signal_prom:.2e}")
        ecdf = bgs.mathematics.empirical_cdf(im_SN.get_array(), signal_prom)
        self._logger.info(
            f"This corresponds to approx. the {ecdf:.4f} quantile of total pixels"
        )
        if np.any(np.abs(prom) > detection):
            self._logger.info(f"Detection of anomolous pixel at {detection}-sigma")
        self._cluster_prom = signal_prom

    def plot_surface_mass_density(self, snap, density_mask, add_extras=True):
        # figure 1: easy, surface mass density
        self._logger.debug("Plotting surface mass density")
        _, _, im, _ = pygad.plotting.image(
            snap.stars[density_mask],
            qty="mass",
            surface_dens=True,
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            cbartitle=r"$\log_{10}\left(\Sigma/\left(\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)\right)$",
            fontsize=self._fontsize,
            outline=None,
            cmap="rocket",
            ax=self.ax[0],
            Npx=self.instrument.number_pixels,
            extent=self.instrument.extent,
        )
        self.ax[0].contour(
            im.get_array(),
            10,
            colors="k",
            linewidths=0.5,
            ls="-",
            extent=im.get_extent(),
            zorder=0.2,
        )
        if add_extras:
            self.add_extras(ax=self.ax[0], pos=snap.bh["pos"])

    def plot_app_magnitude(self, im_mag, snap, add_extras=False):
        self.ax[1].set_facecolor("k")
        im_mag_array = im_mag.get_array()
        im_mag_array = im_mag_array[np.isfinite(im_mag_array)]
        self.ax[1].contour(
            im_mag.get_array(),
            10,
            colors="k",
            linewidths=0.5,
            ls="-",
            extent=im_mag.get_extent(),
            zorder=0.2,
        )
        pygad.plotting.add_cbar(
            ax=self.ax[1],
            cbartitle=r"$\mathrm{mag}\,\mathrm{arcsec}^{-2}$",
            clim=np.percentile(im_mag_array, [0.1, 99.9]),
            cmap=im_mag.get_cmap(),
            fontcolor="w",
            fontsize=self._fontsize,
        )
        pygad.plotting.make_scale_indicators(
            ax=self.ax[1],
            extent=pygad.UnitArr(
                im_mag.get_extent(), units=snap["pos"].units, subs=snap
            ).reshape((2, 2)),
            fontcolor="w",
            fontsize=self._fontsize,
        )
        if add_extras:
            self.add_extras(ax=self.ax[1], pos=snap.bh["pos"])

    def plot_prominence_map(self, im_SN, snap):
        self._logger.debug("Plotting S/N map")
        self.ax[2].set_facecolor("k")
        pygad.plotting.make_scale_indicators(
            ax=self.ax[2],
            extent=pygad.UnitArr(
                im_SN.get_extent(), units=snap["pos"].units, subs=snap
            ).reshape((2, 2)),
            fontcolor="k",
            fontsize=self._fontsize,
        )
        pygad.plotting.add_cbar(
            self.ax[2],
            cbartitle=r"$\mathrm{prominence}$",
            clim=im_SN.get_clim(),
            cmap=im_SN.get_cmap(),
            fontcolor="k",
            fontsize=self._fontsize,
        )


# define the different use cases
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot projected density image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="data",
        type=str,
        help="(list of) snapshot(s) to analyse or plot",
        default="/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0540/output/snap_009.hdf5",
    )
    parser.add_argument(
        "--prominence-only",
        help="only calculate prominences",
        action="store_true",
        dest="prom_only",
    )
    parser.add_argument("-z", dest="redshift", help="redshift", type=float, default=0.6)
    parser.add_argument(
        "--dump", dest="dump", help="dump magnitude data", action="store_true"
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

    if os.path.isfile(args.data):
        # create images for a single snapshot
        # at three different redshifts
        redshifts = [0.2, 0.6, 1.0]
        sigmas = [2.5, 0.5, 0.5]
        SL.warning("Ignoring CL argument '-z'")
        fig, ax = plt.subplots(3, 3)
        fig.set_figwidth(2.5 * fig.get_figwidth())
        fig.set_figheight(2.5 * fig.get_figheight())
        for i, (z, s) in enumerate(zip(redshifts, sigmas)):
            proj_dens = ProjectedDensityObject.load_single_snapshot(
                snapfile=args.data, redshift=z, logger=SL
            )
            proj_dens.ax = ax[i, :]
            proj_dens.prom_scale = 6
            proj_dens.run(dump=args.dump, save_figure=i == 2, sigma=s)
    else:
        data_file = os.path.join(
            figure_config.reduced_data_dir,
            "mag-maps",
            f"{args.data.rstrip('/').split('/')[-2]}-magnitude-maps.pickle",
        )
        try:
            kick_vel_str = re.search("kick-vel-(....)", args.data)
            if kick_vel_str:
                kv = kick_vel_str.group(1)
            else:
                raise RuntimeError(
                    f"Cannot find required kick velocity from data name {args.data}"
                )

            snapnums = bgs.utils.load_data(
                f"/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/perfect_obs/perf_obs_{kv}.pickle"
            )["snapnums"]
            SL.debug(f"Will read snapshots {snapnums}")
        except FileNotFoundError:
            SL.warning(
                "No specific snapshot numbers found based off 'perfect observability'! We will use all snapshots."
            )
            snapnums = None
        proj_dens = ProjectedDensityObject.load_snapshot_list(
            snapdir=args.data,
            redshift=args.redshift,
            saveloc=data_file,
            snapnums=snapnums,
            logger=SL,
        )
        if args.prom_only:
            data = bgs.utils.load_data(data_file)
            for v in data.values():
                proj_dens.calculate_prominence(**v)
        else:
            proj_dens.run()
