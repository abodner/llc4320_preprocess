#!/usr/bin/env python
# coding: utf-8

from xmitgcm import llcreader
model = llcreader.ECCOPortalLLC4320Model()
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime

CASE_NAME = 'california/'
BASE = '/scratch/ab10313/submeso_ML_data/'
PATH = BASE+CASE_NAME


# select subdomain area and downsample every 24 hours to reduce auto-correlation between samples (Uchida et. al. 2022)
# depth above -700m (as we will average over the mixed layer below)
# lat/lon domain to be larger than 12 degrees

lat_min = 30
lat_max = 45
lon_min = -140
lon_max = -125
depth_lim = -700

print('T,S,UV,W FULL DATASETS FROM LLC4320')
print('CASE '+CASE_NAME)
print(datetime.now())
isExist = os.path.exists(PATH)
if isExist:
    print('path already exsits...')
if not isExist:
    os.mkdir(PATH)


isExist = os.path.exists(PATH+'raw_data/')
if isExist:
    print('path already exsits...')
if not isExist:
    os.mkdir(PATH+'raw_data/')

######################################
############## TS ####################
######################################


print('LOAD DATASETS: ds_T,ds_S')
print(datetime.now())
ds_T_full = model.get_dataset(varnames=['Theta'], type='latlon')
ds_S_full = model.get_dataset(varnames=['Salt'], type='latlon')

print('SELECET AREA ON T GRID')
print(datetime.now())
sel_area_T = np.logical_and(np.logical_and(np.logical_and(ds_T_full.XC>lon_min, ds_T_full.XC<lon_max ),
                           np.logical_and(ds_T_full.YC>lat_min, ds_T_full.YC<lat_max)), ds_T_full.Z>depth_lim)

######################################
############## T #####################
######################################

print('ds_T:SELECT')
print(datetime.now())
ds_T = ds_T_full.where(sel_area_T, drop=True).resample(time='24H').nearest(tolerance="1H")

print('ds_T:SAVE NETCDF')
print(datetime.now())
ds_T.to_netcdf(PATH+'raw_data/ds_T.nc',engine='h5netcdf')
del ds_T
print('ds_T:COMPLETE')

######################################
############## S #####################
######################################

print('ds_S:SELECT')
print(datetime.now())
ds_S = ds_S_full.where(sel_area_T, drop=True).resample(time='24H').nearest(tolerance="1H")

print('ds_S:SAVE NETCDF')
print(datetime.now())
ds_S.to_netcdf(PATH+'raw_data/ds_S.nc',engine='h5netcdf')
del ds_S
print('ds_S:COMPLETE')


######################################
############## UV ####################
######################################

print('LOAD DATASET: ds_UV')
print(datetime.now())
ds_UV_full = model.get_dataset(varnames=['U','V'], type='latlon')

######################################
############## U #####################
######################################

print('SELECET AREA ON U GRID')
print(datetime.now())

sel_area_U = np.logical_and(np.logical_and(np.logical_and(ds_UV_full.XG.mean('j_g')>lon_min, ds_UV_full.XG.mean('j_g')<lon_max),
                           np.logical_and(ds_UV_full.YC.mean('i')>lat_min, ds_UV_full.YC.mean('i')<lat_max)),ds_UV_full.Z>depth_lim)

print('ds_U:SELECT')
print(datetime.now())
ds_U = ds_UV_full.U.where(sel_area_U, drop=True).resample(time='24H').nearest(tolerance="1H")

print('ds_U:SAVE NETCDF')
print(datetime.now())
ds_U.to_netcdf(PATH+'raw_data/ds_U.nc',engine='h5netcdf')
del ds_U
print('ds_U:COMPLETE')


######################################
############## V #####################
######################################

print('SELECET AREA ON V GRID')
print(datetime.now())

sel_area_V = np.logical_and(np.logical_and(np.logical_and(ds_UV_full.XC.mean('j')>lon_min, ds_UV_full.XC.mean('j')<lon_max),
                           np.logical_and(ds_UV_full.YG.mean('i_g')>lat_min, ds_UV_full.YG.mean('i_g')<lat_max)),ds_UV_full.Z>depth_lim)

print('ds_V:SELECT')
print(datetime.now())
ds_V = ds_UV_full.V.where(sel_area_V, drop=True).resample(time='24H').nearest(tolerance="1H")

print('ds_V:SAVE NETCDF')
print(datetime.now())
ds_V.to_netcdf(PATH+'raw_data/ds_V.nc',engine='h5netcdf')
del ds_V
print('ds_V:COMPLETE')


######################################
############## W #####################
######################################

print('LOAD DATASET: ds_W')
print(datetime.now())
ds_W_full = model.get_dataset(varnames=['W'], type='latlon')

print('SELECET AREA ON W GRID')
print(datetime.now())

sel_area_W = np.logical_and(np.logical_and(np.logical_and(ds_W_full.XC>lon_min, ds_W_full.XC<lon_max ),
                           np.logical_and(ds_W_full.YC>lat_min, ds_W_full.YC<lat_max)), ds_W_full.Zl>depth_lim)

print('ds_W:SELECT')
print(datetime.now())
ds_W = ds_W_full.where(sel_area_W, drop=True).resample(time='24H').nearest(tolerance="1H")


print('ds_W:SAVE NETCDF')
print(datetime.now())
ds_W.to_netcdf(PATH+'raw_data/ds_W.nc',engine='h5netcdf')
del ds_W
print('ds_W:COMPLETE')







