#!/usr/bin/env python
# coding: utf-8

from xmitgcm import llcreader
model = llcreader.ECCOPortalLLC4320Model()
import numpy as np
import pandas as pd
import xarray as xr
import xrft
import xgcm
import gcm_filters
import matplotlib.pyplot as plt
import xesmf as xe
#import gsw_xarray as gsw
import os
from datetime import datetime
from fastjmd95 import jmd95numba 


CASE_NAME = 'california/'
BASE = '/scratch/ab10313/submeso_ML_data/'
PATH = BASE+CASE_NAME

print('PREPROCESS CASE '+CASE_NAME)
print(datetime.now())


print('CALCULATE OR LOAD B')
print(datetime.now())

# Check whether B already exists
B_path = PATH+'raw_data/B.nc'
sigma0_path = PATH+'raw_data/sigma0.nc'

isExist = os.path.exists(B_path)
if isExist:
    print('B EXISTS: LOADING...')
    B = xr.open_dataarray(B_path,chunks='auto')
    B = B.chunk(chunks={'time': 1, 'k':B.k.size, 'j': B.j.size, 'i': B.i.size})
    sigma0 = xr.open_dataarray(sigma0_path,chunks='auto')
    sigma0 = sigma0.chunk(chunks={'time': 1, 'k':sigma0.k.size, 'j': sigma0.j.size, 'i': sigma0.i.size})
if not isExist:
    print('B DOES NOT EXIST: CALCULATING...')
    print('LOAD ds_T, ds_S AND SET CHUNKS')
    print(datetime.now())
    # load temperature and salinity datasets
    ds_T = xr.open_dataarray(PATH+'raw_data/ds_T.nc',chunks='auto')
    ds_S = xr.open_dataarray(PATH+'raw_data/ds_S.nc',chunks='auto')

    T = ds_T.chunk(chunks={'time': 1, 'k':ds_T.k.size, 'j': ds_T.j.size, 'i': ds_T.i.size})
    S = ds_S.chunk(chunks={'time': 1, 'k':ds_S.k.size, 'j': ds_S.j.size, 'i': ds_S.i.size})

    del ds_T, ds_S

    
    # calculate buoyancy from temp and salt, using the fastjmd95 package
    
    # reference density 
    rho0 = 1000 #kg/m^3

    # potential density anomaly 
    # with the reference pressure of 0 dbar and ρ0 = 1000 kg m−3
    sigma0 = jmd95numba.rho(S, T, 0) - rho0
    sigma0 = sigma0.rename('sigma0')
    
    del T, S

    # gravity
    G = 9.81 #m/s^2

    # buoyancy
    B = -G*sigma0/rho0
    B = B.rename('Buoyancy')

    print('SAVE SIGMA')
    print(datetime.now())
    print('SIGMA:START')
    print(datetime.now())
    sigma0.to_netcdf(sigma0_path,engine='h5netcdf')
    print('SIGMA:COMPLETE')

    print('SAVE B')
    print(datetime.now())
    print('B:START')
    print(datetime.now())
    B.to_netcdf(B_path,engine='h5netcdf')
    print('B:COMPLETE')


print('LOAD OR CALCULATE INTERPOLATED U,V,W')
print(datetime.now())
# Check whether U already exists
U_path = PATH+'raw_data/U.nc'
V_path = PATH+'raw_data/V.nc'
W_path = PATH+'raw_data/W.nc'

isExist = os.path.exists(U_path)
if isExist:
    print('UVW EXIST: LOADING...')
    U = xr.open_dataarray(U_path,chunks='auto')
    U = U.chunk(chunks={'time': 1, 'k':U.k.size, 'j': U.j.size, 'i': U.i.size})
    V = xr.open_dataarray(V_path,chunks='auto')
    V = V.chunk(chunks={'time': 1, 'k':V.k.size, 'j': V.j.size, 'i': V.i.size})
    W = xr.open_dataarray(W_path,chunks='auto')
    W = W.chunk(chunks={'time': 1, 'k':W.k.size, 'j': W.j.size, 'i': W.i.size})
if not isExist:
    print('UVW DO NOT EXIST: CALCULATING...')
    print('LOAD ds_U, ds_V, ds_W AND SET CHUNKS')

    # load full velocity dataset
    ds_U = xr.open_dataarray(PATH+'raw_data/ds_U.nc',chunks='auto')
    ds_V = xr.open_dataarray(PATH+'raw_data/ds_V.nc',chunks='auto')
    ds_W = xr.open_dataarray(PATH+'raw_data/ds_W.nc',chunks='auto')

    # interp full velocities onto tracer grid
    ds_U_interp = ds_U.interp(i_g=B.i,j=B.j)
    ds_V_interp = ds_V.interp(i=B.i,j_g=B.j)
    ds_W_interp = ds_W.interp(k_l=B.k)

    del ds_U, ds_V, ds_W
    
    U = ds_U_interp.chunk(chunks={'time': 1, 'k':ds_U_interp.k.size, 'j': ds_U_interp.j.size, 'i': ds_U_interp.i.size})
    V = ds_V_interp.chunk(chunks={'time': 1, 'k':ds_V_interp.k.size, 'j': ds_V_interp.j.size, 'i': ds_V_interp.i.size})
    W = ds_W_interp.chunk(chunks={'time': 1, 'k':ds_W_interp.k.size, 'j': ds_W_interp.j.size, 'i': ds_W_interp.i.size})

    del ds_U_interp, ds_V_interp, ds_W_interp
   
    print('SAVE UVW')
    print(datetime.now())

    print('U:START')
    print(datetime.now())
    U.to_netcdf(U_path,engine='h5netcdf')
    print('U:COMPLETE')

    print('V:START')
    print(datetime.now())
    V.to_netcdf(V_path,engine='h5netcdf')
    print('V:COMPLETE')

    print('W:START')
    print(datetime.now())
    W.to_netcdf(W_path,engine='h5netcdf')
    print('W:COMPLETE')

# lon lat
lat = B.YC.mean('i')
lon = B.XC.mean('j')

lat_min = lat.min().values
lat_max = lat.max().values
lon_min = lon.min().values
lon_max = lon.max().values

# select middle 10 degree box for snapshot
lon_mid = (lon_min + lon_max)/2
lat_mid = (lat_min + lat_max)/2

lon_slice_10deg = slice(lon_mid-5,lon_mid+5)
lat_slice_10deg = slice(lat_mid-5,lat_mid+5)

sel_area_10deg = np.where(np.logical_and(np.logical_and(B.XC>lon_mid-5, B.XC<lon_mid+5 ),
                           np.logical_and(B.YC>lat_mid-5, B.YC<lat_mid+5)))

i_min_10deg = min(sel_area_10deg[1])
i_max_10deg = max(sel_area_10deg[1])
j_min_10deg = min(sel_area_10deg[0]) 
j_max_10deg = max(sel_area_10deg[0])


i_slice_10deg = slice(i_min_10deg, i_max_10deg)
j_slice_10deg = slice(j_min_10deg, j_max_10deg)

########################
def save_snapshot(PATH, var, var_save_name):

    # snapshot values
    var_values = var.values

    ########## Save  data
    FULL_PATH_SNAP = PATH+'snapshots/'

    # Check whether path exists or not
    isExist = os.path.exists(FULL_PATH_SNAP)
    if not isExist:
        os.makedirs(FULL_PATH_SNAP)
        print("The new snapshots directory is created!")

    np.save(FULL_PATH_SNAP+var_save_name+'.npy', var_values)
    del var_values

#################

save_snapshot(PATH,B.XC.isel(i=i_slice_10deg,j=j_slice_10deg)[0,:],'lon')
save_snapshot(PATH,B.YC.isel(i=i_slice_10deg,j=j_slice_10deg)[:,0],'lat')

save_snapshot(PATH,B.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'B')
save_snapshot(PATH,U.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'U')
save_snapshot(PATH,V.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'V')
save_snapshot(PATH,W.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'W')

print('MLD')
print(datetime.now())
# sigma0 at 10m depth for reference
sigma0_10m = sigma0.isel(k=6).broadcast_like(sigma0).chunk(chunks={'time': 1, 'k':sigma0.k.size, 'j': sigma0.j.size, 'i': sigma0.i.size})

delta_sigma = sigma0 - sigma0_10m

# mixed layer depth
MLD = sigma0.Z.broadcast_like(sigma0).where(delta_sigma<=0.03).min(dim="k",skipna=True).chunk(chunks={'time': 1, 'j': sigma0.j.size, 'i': sigma0.i.size}).rename('Mixed Layer Depth')

save_snapshot(PATH,MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'MLD')

del MLD

print('FILTER')
print(datetime.now())
# # Filtering: fixed factor scale 24

gcm_filters.required_grid_vars(gcm_filters.GridType.IRREGULAR_WITH_LAND)


# define input parameters

# wet mask
wet_mask = xr.DataArray(np.logical_not(np.isnan(B.Depth)).values,coords={'j':B.j.values , 'i': B.i.values})

# area
area= B.rA

#grid


# define grid for filter
dxw = xr.DataArray(
    data=U.dxC, 
    coords={'i':B.i,'j':B.j}, 
    dims=('j','i')
)

dyw = xr.DataArray(
    data=V.dyC, 
    coords={'i':B.i,'j':B.j}, 
    dims=('j','i')
)


dxs = xr.DataArray(
    data=V.dxG, 
    coords={'i':B.i,'j':B.j}, 
    dims=('j','i')
)


dys = xr.DataArray(
    data=U.dyG, 
    coords={'i':B.i,'j':B.j}, 
    dims=('j','i')
)


# dx_min
dx_min = min(dxw.min(),dyw.min(),dxs.min(),dys.min())
dx_min = dx_min.values
dx_min

#dx_max
dx_max = max(dxw.max(),dys.max()) #removed ,dyw.max(),dxs.max()
dx_max = dx_max.values
dx_max

#kappa_w and kappa_s
kappa_w_fac = dxw * dxw / (dx_max * dx_max) 
kappa_s_fac = dys * dys / (dx_max * dx_max)



# define filter
filter_factor24 = gcm_filters.Filter(
    filter_scale=24 * dx_max, # factor of 24
    dx_min=dx_min,
    filter_shape=gcm_filters.FilterShape.GAUSSIAN,
    grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
    grid_vars={
        'wet_mask': wet_mask, 
        'dxw': dxw, 'dyw': dyw, 'dxs': dxs, 'dys': dys, 'area': area, 
        'kappa_w': kappa_w_fac, 'kappa_s': kappa_s_fac
    }
)
filter_factor24


# ## Define Mesoscale and submesoscale fields


# mesoscale field defined by the 24 factor filter

Um = filter_factor24.apply(U.where(wet_mask), dims=['i', 'j'])
Vm = filter_factor24.apply(V.where(wet_mask), dims=['i', 'j'])
Wm = filter_factor24.apply(W.where(wet_mask), dims=['i', 'j'])
Bm = filter_factor24.apply(B.where(wet_mask), dims=['i', 'j'])

print('SAVE SNAPSHOT: Bm,Um,Vm,Wm')
print(datetime.now())
save_snapshot(PATH,Bm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm')
save_snapshot(PATH,Um.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Um')
save_snapshot(PATH,Vm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Vm')
save_snapshot(PATH,Wm.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Wm')

# submesoscale field defined as the deviation between the full resolved field and mesoscale field

Us = U - Um
Vs = V - Vm
Ws = W - Wm
Bs = B - Bm

del U, V, W

print('SAVE SNAPSHOT: Bs,Us,Vs,Ws')
print(datetime.now())
save_snapshot(PATH,Bs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bs')
save_snapshot(PATH,Us.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Us')
save_snapshot(PATH,Vs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Vs')
save_snapshot(PATH,Ws.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Ws')

# submesoscale buoyancy fluxes, with a mesoscale filter

UsBs = filter_factor24.apply((Us * Bs).where(wet_mask), dims=['i', 'j'])
VsBs = filter_factor24.apply((Vs * Bs).where(wet_mask), dims=['i', 'j'])
WsBs = filter_factor24.apply((Ws * Bs).where(wet_mask), dims=['i', 'j'])

del Us, Vs, Ws, Bs

print('SAVE SNAPSHOT: UsBs,VsBs,WsBs')
print(datetime.now())
save_snapshot(PATH,UsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'UsBs')
save_snapshot(PATH,VsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'VsBs')
save_snapshot(PATH,WsBs.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'WsBs')


# mesoscale buoyancy gradient
# need to temperarily rechunk for the z derivative
Bm_x = Bm.differentiate("i")
Bm_y = Bm.differentiate("j")
Bm_z = Bm.differentiate("k")

print('SAVE SNAPSHOT: Bm_x,Bm_y,Bm_z')
print(datetime.now())
save_snapshot(PATH,Bm_x.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_x')
save_snapshot(PATH,Bm_y.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_y')
save_snapshot(PATH,Bm_z.isel(time=0,k=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_z')


print('MLD AVERAGE')
print(datetime.now())
# meso velocities and buoyancy MLD average
Um_MLD = Um.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Um_MLD')
Vm_MLD = Vm.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Vm_MLD')
Wm_MLD = Wm.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Wm_MLD')
Bm_MLD = Bm.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_MLD')
del Um, Vm, Wm, Bm

print('SAVE SNAPSHOT MLD: Bm,Um,Vm,Wm')
print(datetime.now())
save_snapshot(PATH,Bm_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_MLD')
save_snapshot(PATH,Um_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'Um_MLD')
save_snapshot(PATH,Vm_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'Vm_MLD')
save_snapshot(PATH,Wm_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'Wm_MLD')

# submeso fluxes MLD average
UsBs_MLD = UsBs.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('UsBs_MLD')
VsBs_MLD = VsBs.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('VsBs_MLD')
WsBs_MLD = WsBs.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('WsBs_MLD')
del UsBs, VsBs, WsBs


print('SAVE SNAPSHOT MLD: UsBs,VsBs,WsBs')
print(datetime.now())
save_snapshot(PATH,UsBs_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'UsBs_MLD')
save_snapshot(PATH,VsBs_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'VsBs_MLD')
save_snapshot(PATH,WsBs_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'WsBs_MLD')

# meso gradients MLD average
Bm_x_MLD = Bm_x.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_x_MLD')
Bm_y_MLD = Bm_y.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_y_MLD')
Bm_z_MLD = Bm_z.where(delta_sigma<=0.03).mean(dim="k",skipna=True).rename('Bm_z_MLD')
del Bm_x, Bm_y, Bm_z

print('SAVE SNAPSHOT MLD: Bm_x,Bm_y,Bm_z')
print(datetime.now())
save_snapshot(PATH,Bm_x_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_x_MLD')
save_snapshot(PATH,Bm_y_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_y_MLD')
save_snapshot(PATH,Bm_z_MLD.isel(time=0,i=i_slice_10deg,j=j_slice_10deg),'Bm_z_MLD')

print('COARSE-GRAIN')
print(datetime.now())
### Coarse-grain to 1/4 degree
# building regridder
ds_out_qurter = xr.Dataset(
    {
        "lat": (["lat"], np.arange(lat_min, lat_max, 0.25)),
        "lon": (["lon"], np.arange(lon_min, lon_max, 0.25)),
    }
)

regridder_quarter = xe.Regridder(B.isel(time=0,k=0), ds_out_qurter, "bilinear", periodic=False, reuse_weights=False)
regridder_quarter


# regrid and select middle 10 degree box
lon_mid = (lon_min + lon_max)/2
lat_mid = (lat_min + lat_max)/2

lon_slice_10deg = slice(lon_mid-5,lon_mid+5)
lat_slice_10deg = slice(lat_mid-5,lat_mid+5)

# save 10deg lat lowres and lon lowres grids
lat_lowres = np.arange(lat_min, lat_max, 0.25)
lat_lowres = lat_lowres[np.where(np.logical_and(lat_lowres>=lat_mid-5,lat_lowres<=lat_mid+5))]

lon_lowres = np.arange(lon_min, lon_max, 0.25)
lon_lowres = lon_lowres[np.where(np.logical_and(lon_lowres>=lon_mid-5,lon_lowres<=lon_mid+5))]

np.save(PATH+'snapshots/lat_lowres.npy',lon_lowres)
np.save(PATH+'snapshots/lon_lowres.npy',lat_lowres)


# meso velocities and buoyancy
Um_MLD_lowres = regridder_quarter(Um_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Vm_MLD_lowres = regridder_quarter(Vm_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Wm_MLD_lowres = regridder_quarter(Wm_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_MLD_lowres = regridder_quarter(Bm_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
del Um_MLD, Vm_MLD, Wm_MLD, Bm_MLD

# meso gradients
Bm_x_MLD_lowres = regridder_quarter(Bm_x_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_y_MLD_lowres = regridder_quarter(Bm_y_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
Bm_z_MLD_lowres = regridder_quarter(Bm_z_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
del Bm_x_MLD, Bm_y_MLD, Bm_z_MLD

# submeso fluxes
UsBs_MLD_lowres = regridder_quarter(UsBs_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg)  
VsBs_MLD_lowres = regridder_quarter(VsBs_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg) 
WsBs_MLD_lowres = regridder_quarter(WsBs_MLD).sel(lon=lon_slice_10deg,lat=lat_slice_10deg)  
del UsBs_MLD, VsBs_MLD, WsBs_MLD

# SAVE NETCDF
print('SAVE NETCDF: MLD AVG LOWRES')
FULL_PATH_PP = PATH+'preprocessed_data/'
isExist = os.path.exists(FULL_PATH_PP)
if isExist:
    print('path already exsits...')
if not isExist:
    os.mkdir(FULL_PATH_PP)

print('meso fields')
Um_MLD_lowres.to_netcdf(FULL_PATH_PP+'Um_MLD_lowres.nc',engine='h5netcdf')
Vm_MLD_lowres.to_netcdf(FULL_PATH_PP+'Vm_MLD_lowres.nc',engine='h5netcdf')
Wm_MLD_lowres.to_netcdf(FULL_PATH_PP+'Wm_MLD_lowres.nc',engine='h5netcdf')
Bm_MLD_lowres.to_netcdf(FULL_PATH_PP+'Bm_MLD_lowres.nc',engine='h5netcdf')

del Um_MLD_lowres, Vm_MLD_lowres, Wm_MLD_lowres, Bm_MLD_lowres

print('submeso fluxes')
UsBs_MLD_lowres.to_netcdf(FULL_PATH_PP+'UsBs_MLD_lowres.nc',engine='h5netcdf')
VsBs_MLD_lowres.to_netcdf(FULL_PATH_PP+'VsBs_MLD_lowres.nc',engine='h5netcdf')
WsBs_MLD_lowres.to_netcdf(FULL_PATH_PP+'WsBs_MLD_lowres.nc',engine='h5netcdf')

del UsBs_MLD_lowres, VsBs_MLD_lowres, WsBs_MLD_lowres

print('meso B grad')
Bm_x_MLD_lowres.to_netcdf(FULL_PATH_PP+'Bm_x_MLD_lowres.nc',engine='h5netcdf')
Bm_y_MLD_lowres.to_netcdf(FULL_PATH_PP+'Bm_y_MLD_lowres.nc',engine='h5netcdf')
Bm_z_MLD_lowres.to_netcdf(FULL_PATH_PP+'Bm_z_MLD_lowres.nc',engine='h5netcdf')

del Bm_x_MLD_lowres, Bm_y_MLD_lowres, Bm_z_MLD_lowres

print('COMPLETE PREPROCESS')
print(datetime.now())





