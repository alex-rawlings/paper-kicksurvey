import numpy as np

__all__ = ["RecoilCluster", "RecoilClusterSeries"]


class RecoilCluster:
    def __init__(self):
        """
        Class to hold information about the stars in a recoil cluster.
        """
        self.time = None
        self.bh_rad = None
        self.LOS_properties = dict(vel_disp=None, rhalf=None)
        self.intrinsic_properties = dict(vel_disp=None, rhalf=None, bound_mass=None)
        self.ids = None
        self.ambient_vel_disp = None
        self.kick_vel = None
        self.particle_masses = dict(bh=None, stars=None)
        self.snap_num = None

    @property
    def LOS_velocity_dispersion(self):
        return np.linalg.norm(self.LOS_properties["vel_disp"])

    @property
    def effective_radius(self):
        return np.median(self.LOS_properties["rhalf"]) * 1e3

    @property
    def density_3D(self):
        # in Msol / pc^3
        return self.intrinsic_properties["bound_mass"] / (
            4 / 3 * np.pi * (1e3 * self.intrinsic_properties["rhalf"]) ** 3
        )

    @property
    def density_2D(self):
        # in Msol / pc^2
        return self.intrinsic_properties["bound_mass"] / (
            4 * np.pi * self.effective_radius**2
        )


class RecoilClusterSeries:
    def __init__(self, *clusters):
        """
        Class to determine properties of related clusters (i.e. a time-series).

        Parameters
        ----------
        clusters : RecoilCluster
            clusters to add to the series
        """
        self.clusters = clusters
        kick_vels = [c.kick_vel for c in self.clusters]
        assert np.all(np.abs(np.diff(kick_vels)) < 1e-10)
        self.kick_vel = float(kick_vels[0])
        self.snap_nums = [c.snap_num for c in self.clusters]

    @property
    def bh_radii(self):
        """
        Displacement of BHs for each cluster

        Returns
        -------
        : list
            BH displacements
        """
        return [c.bh_rad for c in self.clusters]

    @property
    def max_rad(self):
        """
        Determine the maximum BH displacement

        Returns
        -------
        : float
            maximum radial displacement
        """
        return max(self.bh_radii)

    @property
    def apo(self):
        """
        Determine the cluster corresponding to apocentre.

        Returns
        -------
        : RecoilCluster
            apocentre cluster
        """
        return self.clusters[np.argmax(self.bh_radii)]

    @property
    def peri(self):
        """
        Determine the cluster corresponding to pericentre.

        Returns
        -------
        : RecoilCluster
            pericentre cluster
        """
        return self.clusters[np.argmin(self.bh_radii)]

    @property
    def LOS_velocity_dispersion_near_apo(self):
        """
        Determine LOS velocity dispersion near apocentre by averaging over a
        few snapshots.

        Returns
        -------
        : float
            LOS velocity dispersion
        """
        sigma = np.full((3, 3), np.nan)
        apo_idx = np.argmax(self.bh_radii)
        for i, idx in enumerate((-1, 0, 1)):
            c = self.clusters[apo_idx + idx]
            try:
                assert np.abs(int(c.snap_num) - int(self.apo.snap_num)) < 2
            except AssertionError:
                # raise AssertionError(f"Snapshot numbers are {self.snap_nums} - apocentre is {self.apo.snap_num} - BH radii are {self.bh_radii}")
                print(
                    f"WARNING: not using snapshot {c.snap_num} as it is too far from apocentre"
                )
                continue
            sigma[i, :] = c.LOS_properties["vel_disp"]
        return np.linalg.norm(np.sqrt(np.nanmean(sigma**2, axis=0)))

    @property
    def ambient_sigma_series(self):
        s = []
        for c in self.clusters:
            s.append(c.ambient_vel_disp)
        return s

    @property
    def max_density_3D(self):
        return max([c.density_3D for c in self.clusters])

    @property
    def max_density_2D(self):
        return max([c.density_2D for c in self.clusters])

    def __len__(self):
        return len(self.clusters)
