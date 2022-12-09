# llc4320_preprocess

This repository includes functions to preprocess the llc4320 data.
## llc4320_data 
This is a python code to save 3D variables on a local machine. 
Each variable (Theta, SALT, U, V, W) is defined in the top 700m of a 15X15 degree box and down-sampled to once every 24 hours. 
The example here is for the Califonia Current.

## llc4320_preprocess
This is a python code to preprocess the data to be used in a NN (without normalization). 
A snapshot of each step is saved to help visualize the calculations. 
This includes the following steps:

1. Compute buoyancy from S and T using the fastjmd95 package.
2. Interpolate U,V,W on to the T grid.
3. Calculate the mixed layer depth using the criteria of 0.03 between sigma and its value at 10m.
4. Filter using gcmfilters using a factor scale of 24 that corresponds to 1/4 degree.
5. Define submesoscale fluxes (UsBs, VsBs, WsBs), mesoscale variables (Um, Vm, Wm, Bm), and mesoscale bouyancy gradeints (Bm_x, Bm_y, Bm_z).
6. Average over the mixed layer depth.
6. Coarse-grain to 1/4 degree using the xesmf package.
7. Save netcdfs of each preprocessed variable.
