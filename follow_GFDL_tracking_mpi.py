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
    name=raw_input()
    date=raw_input()
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
    dbz=np.max(flag.variables['REFL_10CM'][0,:,:,:],axis=0)
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
        slp_min=np.min(mslp[cell[0]-6:cell[0]+6,cell[1]-6:cell[1]+6])
        if all(np.asarray(dist)<6.5) and np.max(dbz[cell[0]-6:cell[0]+6,cell[1]-6:cell[1]+6])>=0 and \
            4<lat[cell[0],cell[1]]<40 and 100<lon[cell[0],cell[1]]<150 and xland[cell[0],cell[1]]==2 and \
            wind_max>10:
            possible_loc+=[[lat[cell[0],cell[1]],lon[cell[0],cell[1]]]]
            print date,lat[cell[0],cell[1]],lon[cell[0],cell[1]],wind_max,slp_min/100
    m=Basemap(llcrnrlon=100,llcrnrlat=4,urcrnrlon=150,urcrnrlat=40,projection='mill',resolution='h')
    parallels=np.arange(round(10,1),round(50,1),5)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=12)
    meridians=np.arange(round(70,1),round(180,1),5)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=12)
    m.drawcountries(linewidth=0.5,color='k')
    m.drawcoastlines()
    x,y=m(lon,lat)
    dbz[dbz<=0]=np.nan
    cs=m.scatter(x,y,25,c=dbz,cmap='jet',edgecolors='face')
    cbar=m.colorbar(cs,location='right',pad="5%")
    try:
        mod_loc=np.asarray(possible_loc)
        x,y=m(mod_loc[:,1],mod_loc[:,0])
        m.scatter(x,y,70,c='black')
    except:
        pass
    plt.title(date)
    plt.savefig(date+'.png')
