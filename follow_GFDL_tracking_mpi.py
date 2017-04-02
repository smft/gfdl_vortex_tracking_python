#!/usr/bin/env python

"""author : Qi Zhang """
#######"""NJU"""########

#!/usr/bin/env python

import warnings
import pymongo
import string
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from mpi4py import MPI as MPI
from mpl_toolkits.basemap import Basemap,cm,shiftgrid

def vertical_interp(pressure_data,interp_data,order_level):
    nz,ny,nx=np.shape(pressure_data)
    rslt=np.zeros([ny,nx])
    i=0
    while i<ny:
        j=0
        while j<nx:
            f=(interp1d(pressure_data[:,i,j]/100.0,interp_data[:,i,j],kind='linear',fill_value='extrapolate'))
            rslt[i,j]=f(order_level)
            j+=1
        i+=1
    return rslt

def cal_voricity(u,v):
    dudx,dudy=np.gradient(u)
    dvdx,dvdy=np.gradient(v)
    rslt=dvdx-dudy
    return rslt

def step_1(mslp):
    shp_y,shp_x=np.shape(mslp)
    loc=[]
    min_slp=ndimage.filters.minimum_filter(mslp,size=12)
    ny,nx=np.nonzero(mslp==min_slp)
    i=0
    count=0
    for cell in ny:
        if 6<=ny[i]<=shp_y-6 and 6<=nx[i]<=shp_x-6:
            count+=1
            area=mslp[ny[i]-6:ny[i]+6,nx[i]-6:nx[i]+6]
            area_max=np.max(area)
            idx_y,idx_x=np.nonzero(area==area_max)
            idx_ym,idx_xm=np.nonzero(area==mslp[ny[i],nx[i]])
            dist=np.mean(np.sqrt((idx_y-idx_ym)**2+(idx_x-idx_xm)**2)*18)
            if np.abs((area_max-mslp[ny[i],nx[i]])/dist)/100>=0.0015:
                loc+=[[ny[i],nx[i]]]
        i+=1
    return loc

def step_2(u,v,p,level):
    u_i=vertical_interp(p,u,level)
    v_i=vertical_interp(p,v,level)
    vor=cal_voricity(u_i,v_i)
    max_vor=ndimage.filters.maximum_filter(vor,size=12)
    ny,nx=np.nonzero(vor==max_vor)
    i=0
    loc=[]
    for cell in ny:
        loc+=[[ny[i],nx[i]]]
        i+=1
    return loc

def step_3(hgt,p,level):
    h_i=vertical_interp(p,hgt,level)
    min_hgt=ndimage.filters.minimum_filter(h_i,size=12)
    ny,nx=np.nonzero(h_i==min_hgt)
    i=0
    loc=[]
    for cell in ny:
        loc+=[[ny[i],nx[i]]]
        i+=1
    return loc

"""test!!!test"""
comm=MPI.COMM_WORLD
comm_rank=comm.Get_rank()
comm_size=comm.Get_size()
if comm_rank==0:
    warnings.filterwarnings("ignore")
    bwp_file=raw_input()
    path=raw_input()
    rear=raw_input()
    obs_raw=[cell.split(' ') for cell in bwp_file.split(',')]
    name=path+'/wrfout_d01_'+obs_raw[2][-1][:4]+'-'+obs_raw[2][-1][4:6]+'-'+\
                obs_raw[2][-1][6:8]+'_'+obs_raw[2][-1][8:]+rear
    date=obs_raw[2][-1]
    obs_lat=string.atof(obs_raw[6][-1][:-1])/10
    obs_lon=string.atof(obs_raw[7][-1][:-1])/10
else:
    warnings.filterwarnings("ignore")
    name=None
name=comm.bcast(name,root=0)
if comm_rank==0:
    flag=Dataset(name)
    mslp=flag.variables['PSFC'][0,:,:]
    u10=flag.variables['U10'][0,:,:]
    v10=flag.variables['V10'][0,:,:]
    xland=flag.variables['XLAND'][0,:,:]
    lat=flag.variables['XLAT'][0,:,:]
    lon=flag.variables['XLONG'][0,:,:]
    raw_loc=step_1(mslp)
    vor10=cal_voricity(u10,v10)
    max_vor10=ndimage.filters.maximum_filter(vor10,size=12)
    ny,nx=np.nonzero(vor10==max_vor10)
    loc=[]
    for cell in raw_loc:
        dist=np.sqrt((ny-cell[0])**2+(nx-cell[1])**2)
        if np.min(dist)<6.5:
            loc+=[cell]
    loc=np.asarray(loc)
elif comm_rank==1:
    flag=Dataset(name)
    u=flag.variables['U'][0,:,:,:]
    v=flag.variables['V'][0,:,:,:]
    p=(flag.variables['P'][0,:,:,:]+flag.variables['PB'][0,:,:,:])/100
    loc=np.asarray(step_2(u,v,p,850))
elif comm_rank==2:
    flag=Dataset(name)
    u=flag.variables['U'][0,:,:,:]
    v=flag.variables['V'][0,:,:,:]
    p=(flag.variables['P'][0,:,:,:]+flag.variables['PB'][0,:,:,:])/100
    loc=np.asarray(step_2(u,v,p,700))
elif comm_rank==3:
    flag=Dataset(name)
    hgt=(flag.variables['PH'][0,:,:,:]+flag.variables['PHB'][0,:,:,:])
    p=(flag.variables['P'][0,:,:,:]+flag.variables['PB'][0,:,:,:])/100
    loc=np.asarray(step_3(hgt[1:,:,:],p,850))
elif comm_rank==4:
    flag=Dataset(name)
    hgt=(flag.variables['PH'][0,:,:,:]+flag.variables['PHB'][0,:,:,:])
    p=(flag.variables['P'][0,:,:,:]+flag.variables['PB'][0,:,:,:])/100
    loc=np.asarray(step_3(hgt[1:,:,:],p,700))
loc_all=comm.gather(loc,root=0)
if comm_rank==0:
    possible_loc=[]
    for cell in loc_all[0]:
        dist=[np.min(np.sqrt((loc_all[1][:,0]-cell[0])**2+(loc_all[1][:,1]-cell[1])**2)),\
                np.min(np.sqrt((loc_all[2][:,0]-cell[0])**2+(loc_all[2][:,1]-cell[1])**2)),\
                np.min(np.sqrt((loc_all[3][:,0]-cell[0])**2+(loc_all[3][:,1]-cell[1])**2)),\
                np.min(np.sqrt((loc_all[4][:,0]-cell[0])**2+(loc_all[4][:,1]-cell[1])**2))]
        wind=np.sqrt(u10**2+v10**2)
        wind_max=np.max(wind[cell[0]-6:cell[0]+6,cell[1]-6:cell[1]+6])
        slp_min=np.min(mslp[cell[0]-6:cell[0]+6,cell[1]-6:cell[1]+6])/100
        if all(np.asarray(dist)<6.5):
            possible_loc+=[[lat[cell[0],cell[1]],lon[cell[0],cell[1]],wind_max,slp_min]]
    distance=np.sqrt((np.asarray(possible_loc)[:,0]-obs_lat)**2+\
                        (np.asarray(possible_loc)[:,1]-obs_lon)**2)
    idx=np.unravel_index(distance.argmin(),distance.shape)[0]
    print date,possible_loc[idx][0],possible_loc[idx][1],possible_loc[idx][2],possible_loc[idx][3]
