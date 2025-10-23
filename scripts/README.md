# Information on scripts for this project  

These directories are rarely well organised, as the figures that end up being in the paper rarely are exactly the ones that were planned.

## Assumed constants  
From the first study, we use:
- velocity dispersion of the core: 270 km/s  
- core radius due to binary scouring: 0.58 kpc (median value)

## Script dependicies 

### apocentres_and_angle.py  
- requires a threshold detection function, which can be obtained from analysing the output of `proj_density.py` run with extract mode  

### bound_stars.py  
- requires the number of bound particles per snapshot, which has been calculated with `code/misc/recoil-explore/extract_photo_kin.py`. Probably should just write a simple script that gets the number of bound particles without all the IFU stuff and put in the `scripts` directory...  

### compare_compacts.py  
- requires the cluster properties, determined using `perfect_observability.py`  

### ifu_times.py
- no required dependicies  
- does plot IFU maps at four given times, which may require the script to be run a few times, or output from `code/misc/recoil-explore/extract_photo_kin.py` to get the optimal snapshots  

### perfect_observability.py  
- no dependencies  

### proj_density.py  
- no dependicies  