from itertools import chain
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config


muse_files = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/ifu/muse_ifu_mock_02.pickle",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/minor_mergers/ifu/muse_ifu_mock_minor_02.pickle",
]
data_major = bgs.utils.load_data(muse_files[0])
data_minor = bgs.utils.load_data(muse_files[1])


# set up generators
def _vor_generator(d, yield_t):
    for v, t in zip(d["voronoi"][1:], d["t"][1:]):
        if yield_t:
            yield v, t
        else:
            yield v


# get the colour limits
def vor_generator_M(yield_t=False):
    vg = _vor_generator(data_major["0540"], yield_t=yield_t)
    for _vg in vg:
        yield _vg


def vor_generator_m(yield_t=False):
    vg = _vor_generator(data_minor["0000"], yield_t=yield_t)
    for _vg in vg:
        yield _vg


vor_gen_M = vor_generator_M()
vor_gen_m = vor_generator_m()
clims_M = bgs.analysis.unify_IFU_colour_scheme(vor_gen_M)
clims_m = bgs.analysis.unify_IFU_colour_scheme(vor_gen_m)

cbar_kwargs = {"labelsize": 6}
vor_gen_M = vor_generator_M(yield_t=True)
vor_gen_m = vor_generator_m(yield_t=True)

fig, ax = plt.subplot_mosaic(
    """
    AB.
    CD.
    ...
    EF.
    GH.
    """,
    height_ratios=[1, 1, 0.2, 1, 1],
    width_ratios=[1, 1, 0.2],
)
fig.set_figheight(1.5 * fig.get_figheight())
fig.set_figwidth(1.2 * fig.get_figwidth())
fig.text(
    0.03,
    0.72,
    r"$\mathrm{major}$",
    ha="center",
    fontsize="x-large",
    rotation="vertical",
)
fig.text(
    0.03,
    0.20,
    r"$\mathrm{minor}$",
    ha="center",
    fontsize="x-large",
    rotation="vertical",
)

for i, ((vg, t), axsig, axh3) in enumerate(
    zip(chain(vor_gen_M, vor_gen_m), "ACEG", "BDFH")
):
    voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(vg)

    clims = clims_M if axsig in "AC" else clims_m
    voronoi.plot_kinematic_maps(
        ax=ax[axsig], moments="2", clims=clims, cbar="adj", cbar_kwargs=cbar_kwargs
    )
    voronoi.plot_kinematic_maps(
        ax=ax[axh3], moments="3", clims=clims, cbar="adj", cbar_kwargs=cbar_kwargs
    )
    for axi in (ax[axsig], ax[axh3]):
        axi.set_xticks([])
        axi.set_yticks([])
        bgs.plotting.draw_sizebar(ax=axi, length=10, units="kpc", size_vertical=0.5)
    ax[axsig].text(
        0.05, 0.9, f"${t:.3f}\,\mathrm{{Gyr}}$", transform=ax[axsig].transAxes
    )

bgs.plotting.savefig(figure_config.fig_path("IFU_minor_comp.pdf"), force_ext=True)
