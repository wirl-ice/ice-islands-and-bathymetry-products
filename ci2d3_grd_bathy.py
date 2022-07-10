#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci2d3_bathy.py

Script to compare gridded bathymetric products vs an estimated keel depth for ice islands. 

Created on June 18 26 15:29:58 2021

@author: dmueller
"""

# import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
import numpy as np


# functions
def thick2draft(thickness, rho_i = 917, rho_w = 1000):
    """
    calculate ice draft below sea level given the ice thicknes
    assuming hydrostatic equilibrium and tabular profile
  
    *INPUT*
    thickness : the ice thickness in metres
    rho_i : the density of ice (kg/m3)
    rho_w : the density of water (kg/m3)
    
    *OUTPUT*
    draft : draft of ice (depth in m below water level as a positive number)
    
    """
    return thickness*rho_i/rho_w  

def imgclip(img_in, img_out, polygon, nodata=999, maskout=True):
    """
    Function that crops img_in to poly and results in a geotiff 

    *INPUT*
    img_in : the name and ext of the image you want to clip from
    img_out : the name and ext (.tif) of the geotif to save in the current dir
    polygon : a geopandas object of the polygon you are working on (in image proj)
    maskout  : do you want the resultant image to mask data outside the polygon?
        (yes is true - pixels outside the polygon will be nodata)

    *OUTPUT*
    img_out : basename of a tif file
    array   : numpy array of the clipped img
    
    """
    
    # open source image read only
    lc = rio.open(img_in)

    # ask what no data is and if none, make it something
    nd = lc.nodata
    if nd is None:
        nd = nodata

    # clip the raster to that area
    if maskout:
        lc_clip, lc_affine = mask(lc, polygon.geometry, all_touched=True, crop=True, filled=True, nodata=nd)
    else:
        lc_clip, lc_affine = mask(lc, polygon.geometry, all_touched=True,  crop=True, filled=False, nodata=nd)
    lc.close()  # close the dataset
    
    # now replace the no data values from the clipped image so all no data can be the same: 
    lc_clip[lc_clip == nd] = nodata

    clip_lc_meta = lc.meta.copy()
    clip_lc_meta.update({"height" : lc_clip.shape[1],
                         "width" : lc_clip.shape[2],
                         'nodata': nodata,
                         "transform" : lc_affine
                             })
    ##write the image
    with rio.open(img_out+".tif", "w", **clip_lc_meta) as dest:
        dest.write(lc_clip)
    return lc_clip

def assignNickname(df, grddf, xn=1):
    '''
    Assign a nickname to each entry in a dataframe (df) based on another dataframe
    with 2 columns - a nickname and a list of instances (grdlist)

    Parameters
    ----------
    df : geopandas df
        ice island dataframe
    grdlist : pd dataframe
        2 columns - nickname and a list of instances
    xn :  int
        number of observations needed to qualify
        xn = 1 means 1 observation is required to qualify as a 'grounding', 2 
            means need 2... 

    Returns
    -------
    a geodataframe with new col nickname

    '''
    df.insert(df.shape[1], 'nickname',"999")
    df.insert(df.shape[1], 'firstgrnd',False)
    for i in range(len(grddf)):
        #pull out the nickname
        nickname = grddf.iloc[i,0]
        # pull out the lineage
        lineage = grddf.iloc[i,1]
        #strip quotes and split to make list 
        lineage = [inst.strip('\'') for inst in lineage[1:-1].strip('][').split(', ')]
        lineage.sort() # sort to make sure in correct order
        # If observations of a grounding are not long enough then change to 000
        if len(lineage)<xn:
            for inst in lineage:
                df.loc[df.inst == inst,'nickname'] = "000"
                if inst == lineage[0]:
                    df.loc[df.inst == inst,'firstgrnd'] = True
        else:
            for inst in lineage:
                df.loc[df.inst == inst,'nickname'] = nickname
                if inst == lineage[0]:
                    df.loc[df.inst == inst,'firstgrnd'] = True
    df = df.reset_index(drop=True)
    # these are unassiged
    n1=len(df.loc[df.nickname == '999'])
    if n1 > 0:
        print(f"There are {n1} ice islands that are unnamed, check")
    n2 = len(    df.loc[((df['nickname'] == '000') & (df['firstgrnd'] == True))] )
    print(f"There are {n2} ice island groundings with fewer than {xn} observations")
    print("These ice islands were removed from the analysis")
    # remove these
    df = df.loc[df.nickname != "999"]
    df = df.loc[df.nickname != "000"]
    return df
                               

''' MAIN '''        
# parameters

run = 'sd6' # enter the name of the run (label how you wish)
description = "Run based on calculation of Kd with mean +/- 6 x SD"  # enter notes
sd = 6   # the number of standard deviations for max draft estimate.

#run = 'mm' # enter the name of the run (label how you wish)
#description = "Run based on the Muenchow max thickness"  # enter notes
#sd = 0   # the number of standard deviations for max draft estimate.

#run = 'mm50' # enter the name of the run (label how you wish)
#description = "Run based on the Muenchow max thickness x 50%"  # enter notes
#sd = 0   # the number of standard deviations for max draft estimate.


# if you want to specify the Kd (max keel depth) for a given calving year, that can be done here
# change these from 0 to whatever value you wish (negative numbers for Kd)
specify_Kd_2008 = 0
specify_Kd_2010 = 0
specify_Kd_2012 = 0
#specify_Kd_2010 = -1*thick2draft(108, rho_i=917, rho_w=1025)
#specify_Kd_2012 = -1*thick2draft(228, rho_i=917, rho_w=1025)
#specify_Kd_2010 = -1*thick2draft(108*1.5, rho_i=917, rho_w=1025)
#specify_Kd_2012 = -1*thick2draft(228*1.5, rho_i=917, rho_w=1025)

# Files and directories
indir = '<name of working dir>'  # directory that contains all input files
outdir = f'<name of working dir>/{run}'  # directory for script output

ci2d3 = 'CI2D3_v01.1_selected.shp'     # shapefile containing ci2d3 data
lineagefile = 'grounded_lineage.csv'  # a csv with all the names of ice islands that are grounded (more than once)
ibcao_img = "IBCAO_v4_200m.tif"      # the name of the IBCAO data file (geotiff)
tid_img = "IBCAO_v4_200m_TID.tif"    # the name of the IBCAO TID file (geotiff)
sid_img = "IBCAO_v4_200m_SID.tif"    # the name of the IBCAO SID file (geotiff) - optional in practice
etopo1_img = "ETOPO1_Bed_g_geotiff.tif"   # The name of the ETOPO1 file
ibcao_crs = 'EPSG:3996'   #EPSG code for the IBCAO projection
etopo_crs = 'EPSG:4326'   #EPSG code for the ETOPO1 projection

os.chdir(indir)

print(f'---------Ice island grounding location - Run: {run} ---------\n\n')
print(f'{description} \n')

######  VECTOR DATA
ii = gpd.read_file(ci2d3)
print(f"Reading in {len(ii)} ice island observations from the CI2D3 database")
#select only Petermann Gl. for calving episodes 2008, 2010 and 2012 - 
ii = ii[(ii.calvingloc == 'PG') & ((ii.calvingyr == '2008') | (ii.calvingyr == '2010') | (ii.calvingyr == '2012') )]
print(f"Only {len(ii)} ice island observations originated from Petermann Gl in the calving years 2008, 2010 and 2012")

# select only grounded ice islands
ii = ii.loc[ii.ddinfo == 'grounded']
print(f"{len(ii)} were grounded observations")

# Remove any georef >400m
ii = ii.loc[ii['georef'] != '>400m']
print(f"{len(ii)} had georeferencing better than 400 m accuracy, and were retained\n")

grdlist = pd.read_csv(lineagefile)
print(f"There are {len(grdlist)} grounding events to examine")

ii = assignNickname(ii, grdlist, xn=2)  #Note there are 2 unassigned grounded ice islands UVE and OAQ (and that's ok)

# must reset the index after any selection you do so the concatentation goes well below. 
ii = ii.reset_index(drop=True)
print(f"There are {ii.shape[0]} ice island polygons to examine in a total of {len(ii.nickname.unique())} grounding events\n")

# Thickness and draft
thick2008_mean = 62.3 #m  based on draft calculated with rho_i = 873, rho_w = 1025 (Crawford 2018 JGR)
thick2008_sd = 6.8
# This back-calculates the draft to Humphrey's data
Kd_2008 = -1*thick2draft(thick2008_mean+sd*thick2008_sd, rho_i=873, rho_w=1025)

thick2010_mean = 77.6 #m  based on radar and alitmetry (Munchow pers comm 2022)  "I find 77.6+/15.1 m for the 2010 segment (N=538)...." 
thick2010_sd = 15.1
# conservative estimate of draft using thickest ice - note different densities
Kd_2010 = -1*thick2draft(thick2010_mean+sd*thick2010_sd, rho_i=917, rho_w=1025)

thick2012_mean = 182.7 #m  based on radar and alitmetry (Munchow et al 2014)  "... and 182.7+/-15.2 m for the 2012 segment (N=285)."
thick2012_sd = 15.2
# conservative estimate of draft using thickest ice - note different densities
Kd_2012 = -1*thick2draft(thick2012_mean+sd*thick2012_sd, rho_i=917, rho_w=1025)

print('Ice island thickness:')
print("Note this is based on the parameter sd and may not be realistic due to non-normal distributions (e.g., negative thickness")
print(" Thickness in 2008 ranges from : {0:.1f} to {1:.1f} m".format(thick2008_mean-sd*thick2008_sd, thick2008_mean+sd*thick2008_sd) )
print(" Thickness in 2010 ranges from : {0:.1f} to {1:.1f} m".format(thick2010_mean-sd*thick2010_sd, thick2010_mean+sd*thick2012_sd) )
print(" Thickness in 2012 ranges from : {0:.1f} to {1:.1f} m".format(thick2012_mean-sd*thick2012_sd, thick2012_mean+sd*thick2012_sd) )

print('Ice island estimated maximum draft (based on the parameter sd):')
print(" Kd in 2008 is  : {0:.1f} m".format(Kd_2008) )
print(" Kd in 2010 is  : {0:.1f} m".format(Kd_2010) )
print(" Kd in 2012 is  : {0:.1f}  m".format(Kd_2012) )
print("\n NOTE: Kd is represented as an elevation wrt to water level (as is bathymetry) - i.e., negative numbers")

# Check if the maxdraftYYY inputs are 0 or not.  If they are not zero then 
if specify_Kd_2008 != 0:
    print('Kd for 2008 was specified as {0:.1f}'.format(specify_Kd_2008))
    Kd_2008 = specify_Kd_2008
if specify_Kd_2010 != 0:
    print('Kd for 2010 was specified as {0:.1f}'.format(specify_Kd_2010))
    Kd_2010 = specify_Kd_2010
if specify_Kd_2012 != 0:
    print('Kd for 2012 was specified as {0:.1f}'.format(specify_Kd_2012))
    Kd_2012 = specify_Kd_2012

ii['Kd'] = Kd_2008
ii.loc[ ii.calvingyr == '2010','Kd'] = Kd_2010
ii.loc[ii.calvingyr == '2012','Kd'] = Kd_2012 

#reproject to IBCAO projection
ii_t = ii.to_crs(ibcao_crs)
#sum(ii_t.area)

#buffer on the geolocation error to be conservative.  
ii_t.loc[ii_t.georef =="0-100m", 'geometry'] = ii_t.loc[ii_t.georef =="0-100m"].buffer(100)
ii_t.loc[ii_t.georef =="100-200m",'geometry'] = ii_t.loc[ii_t.georef =="100-200m"].buffer(200)
ii_t.loc[ii_t.georef =="200-400m",'geometry'] = ii_t.loc[ii_t.georef =="200-400m"].buffer(400)
ii_t['area'] = ii_t.area

# copy dataframe to new dataframe for grounded only:
grd = ii_t

#Note I checked etopo ones below 64N by 'hand' and there was nothing to report
#grd.lat[grd.lat<65]   # there are 4 instances but Kd is not higher than Bshall

# this has ibcao and etopo data
grd = grd.loc[grd.lat>64]  
grd = grd.reset_index(drop=True)

# add all these new columns (with bad data flags or default values)

grd.insert(grd.shape[1], "Bshall",999.0)  # shallowest bathymetry under ice island - used to flag issues - compare w/ Kd
grd.insert(grd.shape[1], "Bdeep",999.0)  # deepest bathymetry under ice island (not super important) 
grd.insert(grd.shape[1], "Bmean",999.0) # mean bathymetry under ice island (not as important)

#See GEBCO TID Grid coding  
# number of pixels by TID
grd.insert(grd.shape[1], "P_n",-999)  # number of pixels for all
grd.insert(grd.shape[1], "P_direct",-999)  # TIDs between 10 and 19 - assumed to be very acccuate
grd.insert(grd.shape[1], "P_indir",-999)  # TIDs in 40s  - assumed to be interpolations
grd.insert(grd.shape[1], "P_unkn",-999)   # all unknowns is TID 70 which corresponds to NONNA data
grd.insert(grd.shape[1], "reliablefr",-999.0)# fraction of all pixels that are direct+unknown 

### clearance is Bshall - Kd where TID is direct, indirect, unknown
grd.insert(grd.shape[1], "C",-999.0)  #  Clearance for all data types
grd.insert(grd.shape[1], "C_direct",-999.0)  # TIDs between 10 and 19 - assumed to be very acccuate
grd.insert(grd.shape[1], "C_indir",-999.0)  # TIDs in 40s  - assumed to be interpolations
grd.insert(grd.shape[1], "C_unkn",-999.0)   # all unknowns is TID 70 which corresponds to NONNA data


### More info about Bshall
grd.insert(grd.shape[1], "Bshall_TID",999)  # TID of the shallowest point 
grd.insert(grd.shape[1], "hi_ind_pix",999)  # Number of indirect pixels above highest reliable (direct & unknown)

## Etopo data
grd.insert(grd.shape[1], "eBshall",999.0)
grd.insert(grd.shape[1], "eBdeep",999.0)
grd.insert(grd.shape[1], "eBmean",999.0)
grd.insert(grd.shape[1], "eP_n",-999)
grd.insert(grd.shape[1], "eC",-999)

## Issue flags
grd.insert(grd.shape[1], "issueibcao",0)
grd.insert(grd.shape[1], "issueetopo",0)

if not os.path.isdir(outdir):
    os.mkdir(outdir)

# note that results will be overwritten!
os.chdir(outdir)

for i in range(len(grd)):
    # go through each row in the dataframe
    iceisl = grd.iloc[[i]]
    
    '''save images and extract numpy array of the BUFFERED ice island polygon
    # b - bathymetry IBCAO
    # t - type IBCAO
    # e - bathymetry etopo
    '''
    
    b = imgclip(os.path.join(indir,ibcao_img), '{}_ibcao_b'.format(iceisl.inst.values[0]), iceisl)
    t = imgclip(os.path.join(indir,tid_img), '{}_ibcao_t'.format(iceisl.inst.values[0]), iceisl).astype(np.float32)
    e = imgclip(os.path.join(indir,etopo1_img), '{}_etopo_b'.format(iceisl.inst.values[0]), iceisl.to_crs(etopo_crs) ).astype(np.float32)

    # convert no data to nan. (need to be float first)
    b[b==999] = np.nan
    t[t==999] = np.nan
    e[e==999] = np.nan

    vals, counts = np.unique(t[np.logical_not(np.isnan(t))], return_counts = True)
    # check here to make sure that the counts are the same
    if counts.sum() != b[np.logical_not(np.isnan(b))].size:
        print('error in count for {}'.format(iceisl.inst.values[0]))
        
    grd.iloc[i,grd.columns.get_loc('Bshall')] = np.nanmax(b)
    grd.iloc[i,grd.columns.get_loc('Bdeep')] = np.nanmin(b)
    grd.iloc[i,grd.columns.get_loc('Bmean')] = np.nanmean(b)
    
    grd.iloc[i,grd.columns.get_loc('P_n')] = b[np.logical_not(np.isnan(b))].size
    grd.iloc[i,grd.columns.get_loc('P_direct')]  = counts[vals<20].sum()   # NOTE ASSUMING THERE ARE NO TIDs of 0
    grd.iloc[i,grd.columns.get_loc('P_indir')] = counts[vals<50].sum()-counts[vals<20].sum()
    grd.iloc[i,grd.columns.get_loc('P_unkn')]  = counts[vals>50].sum()
    grd.iloc[i,grd.columns.get_loc('reliablefr')] = ((counts[vals>50].sum()+counts[vals<20].sum())/b[np.logical_not(np.isnan(b))].size)

    grd.iloc[i,grd.columns.get_loc('C')]  = iceisl.Kd - np.nanmax(b)
    grd.iloc[i,grd.columns.get_loc('C_direct')]  = np.nan if np.where(t<20)[0].size ==0 else iceisl.Kd - b[np.where(t<20)].max()
    grd.iloc[i,grd.columns.get_loc('C_indir')] = np.nan if np.where((t>20) & (t<70))[0].size ==0 else iceisl.Kd - b[np.where((t>20) & (t<70))].max() 
    grd.iloc[i,grd.columns.get_loc('C_unkn')]  = np.nan if np.where(t>50)[0].size ==0 else iceisl.Kd - b[np.where(t>50)].max()

    # TID of the shallowest point
    grd.iloc[i,grd.columns.get_loc('Bshall_TID')] = t[np.where(b==np.nanmax(b))].min()  # in case of multiple returns take min
    # Number of indirect pixels above highest reliable (direct/unknown)
    try: 
        # There are numerous issues that could go wrong - no indirect for example
        b_ind = b[np.where((t>20) & (t<70))]   #bathy from indirect TID only
        #Highest reliable: 
        rBshall = b_ind.min() if np.where((t>50) | (t<20)) else b[np.where((t>50) | (t<20))].max()
        hi_ind_pix = np.where(b_ind>rBshall)[0].size
    except: 
        hi_ind_pix = np.nan
    grd.iloc[i,grd.columns.get_loc('hi_ind_pix')]  = hi_ind_pix
    
    # Note that sometimes you get an 'all nan' slice so this means the etopo is mucked up    
    grd.iloc[i,grd.columns.get_loc('eBshall')] = np.nanmax(e)
    grd.iloc[i,grd.columns.get_loc('eBdeep')] = np.nanmin(e)
    grd.iloc[i,grd.columns.get_loc('eBmean')] = np.nanmean(e)
    grd.iloc[i,grd.columns.get_loc('eP_n')] = e[np.logical_not(np.isnan(e))].size
    grd.iloc[i,grd.columns.get_loc('eC')]  = iceisl.Kd - np.nanmax(e)
  
  
    # if this is true, we have an issue with ibcao or etopo bathy data
    if iceisl.Kd.values[0] > np.nanmax(b):
        grd.iloc[i,grd.columns.get_loc('issueibcao')] = 1
    if iceisl.Kd.values[0] > np.nanmax(e):
        grd.iloc[i,grd.columns.get_loc('issueetopo')] = 1
        
    # no sense keeping these images if there are no issues    
    if grd.iloc[i,grd.columns.get_loc('issueibcao')] == 0: 
        os.remove('{}_ibcao_b.tif'.format(iceisl.inst.values[0]))
        os.remove('{}_ibcao_t.tif'.format(iceisl.inst.values[0]))
    if grd.iloc[i,grd.columns.get_loc('issueetopo')] == 0:
        os.remove('{}_etopo_b.tif'.format(iceisl.inst.values[0]))  

#test to see if you have bad values   
if (grd['C'] ==-999.0).any():
    print("Some Clearance values were not calculated")
if (grd['C_direct'] ==-999.0).any():
    print("Some Clearance_direct values were not calculated")
if (grd['C_indir'] ==-999.0).any():
    print("Some Clearance_indirect values were not calculated")
if (grd['C_unkn'] ==-999.0).any():
    print("Some Clearance_unknown values were not calculated")


# make all negative clearance values nan. 
grd.loc[ grd['C'] <0, 'C'] = np.nan
grd.loc[grd['C_direct'] <0,'C_direct'] = np.nan
grd.loc[grd['C_indir'] <0, 'C_indir'] = np.nan
grd.loc[grd['eC'] <0, 'eC'] = np.nan

########-------#####  Table 1
# export data table - one row for each grounded ice island 
# rounding for legibility in exported table
grd1 = grd.round(decimals=4)

# uncomment for exporting polygons with IBCAO or ETOPO issues ONLY
grd1 = grd1.loc[(grd1['issueibcao'] + grd1['issueetopo']) >0]
grd1 = grd1.sort_values(['nickname','inst'])
grd1.to_file(f'ci2d3_grd_ii_{run}.shp')
grd2=grd1.loc[:, grd1.columns != 'geometry']
grd2.to_csv(f'ci2d3_grd_ii_{run}.csv', index=False)


########-------#####  Table 2a
### export summary table for IBCAO
# group and count by issue
grdgrp = grd.groupby(['nickname', 'issueibcao'])['nickname'].agg('count')

# move index to column
grdgrp = grdgrp.reset_index(level=[1])
# only want groups with issues
grdgrp = grdgrp.loc[grdgrp.issueibcao == 1]

# select from the detailed dataframe - the next line gets all grounded polygons 
# that have issues but also the polygons in that grounding that don't have any 
# issues in that event
grd_summary = grd[grd['nickname'].isin(grdgrp.index)]

# Now narrow down again!  This line of code will only select the ibcao issues 
# in the summary, comment it to leave in all polygons
grd_summary = grd_summary.loc[grd_summary.issueibcao == 1]

# set TID here to 1 for reliable and 0 not reliable
grd_summary['Bshall_TID'] = grd_summary['Bshall_TID'].astype(int)
grd_summary.loc[grd_summary['Bshall_TID'] >= 70, 'Bshall_TID'] = 1
grd_summary.loc[grd_summary['Bshall_TID'] <= 20, 'Bshall_TID'] = 1
grd_summary.loc[grd_summary['Bshall_TID'] > 20, 'Bshall_TID'] = 0

# need to get stats for all events
grd_events = grd_summary.groupby(['nickname']).agg(
    count = ('inst','count'),
    first_inst =('inst', 'first'),
    geometry =('geometry', 'first'),
    lat=('lat', 'first'),
    lon=('lon', 'first'),
    Kd =('Kd', 'mean'),
    Bshall_mx =('Bshall', 'max'),
    Bshall_av =('Bshall', 'mean'),
    Bshall_mn =('Bshall', 'min'),
    rel_mx =('reliablefr', 'max'),
    rel_av =('reliablefr', 'mean'),
    rel_mn =('reliablefr', 'min'),
    C_dir_mx=('C_direct', 'max'),  # the max of all clearances  
    C_ind_mx=('C_indir', 'max'), 
    C_unk_mx=('C_unkn', 'max'),
    C_dir_av=('C_direct', 'mean'),  # the mean of all clearances
    C_ind_av=('C_indir', 'mean'), 
    C_unk_av=('C_unkn', 'mean'),
    C_dir_mn=('C_direct', 'min'), # the min of all clearances
    C_ind_mn=('C_indir', 'min'), 
    C_unk_mn=('C_unkn', 'min'),
    BsTID_rel=('Bshall_TID', 'sum')
    )

grd_events['BsTID_rel'] = grd_events['BsTID_rel']/grd_events['count']

grd_events = grd_events.round(decimals=3)

# convert to gpd dataframe
grd_events = gpd.GeoDataFrame(grd_events, geometry='geometry')    
grd_events = grd_events.sort_values(['nickname'])

grd_events.to_file('ci2d3_grd_events_ibcao_{}.shp'.format(run))
grd_events=grd_events.loc[:, grd_events.columns != 'geometry']    
grd_events.to_csv('ci2d3_grd_events_ibcao_{}.csv'.format(run))

print("\nSummary of run: \n")
print(f"Total grounding events with IBCAO issues: {len(grd_events)}")

print(f"{len(grd)} ice island polygons were analyzed (based on selection criteria)")
print(f"{sum(grd.issueibcao)} ice island polygons ({round(sum(grd.issueibcao)/len(grd)*100,2)} %) had a potential issue with ibcao") 

########-------#####  Table 2b
### export summary table for ETOPO
# group and count by issue
grdgrp = grd.groupby(['nickname', 'issueetopo'])['nickname'].agg('count')

# move index to column
grdgrp = grdgrp.reset_index(level=[1])
# only want groups with issues
grdgrp = grdgrp.loc[grdgrp.issueetopo == 1]

# select from the detailed dataframe - the next line gets all grounded polygons 
# that have issues but also the polygons in that grounding that don't have any 
# issues in that event
grd_summary = grd[grd['nickname'].isin(grdgrp.index)]

# Now narrow down again!  This line of code will only select the ibcao issues 
# in the summary, comment it to leave in all polygons
grd_summary = grd_summary.loc[grd_summary.issueetopo == 1]

# set TID here to 1 for reliable and 0 not reliable
grd_summary['Bshall_TID'] = grd_summary['Bshall_TID'].astype(int)
grd_summary.loc[grd_summary['Bshall_TID'] >= 70, 'Bshall_TID'] = 1
grd_summary.loc[grd_summary['Bshall_TID'] <= 20, 'Bshall_TID'] = 1
grd_summary.loc[grd_summary['Bshall_TID'] > 20, 'Bshall_TID'] = 0

# need to get stats for all events
grd_events = grd_summary.groupby(['nickname']).agg(
    count = ('inst','count'),
    first_inst =('inst', 'first'),
    geometry =('geometry', 'first'),
    lat=('lat', 'first'),
    lon=('lon', 'first'),
    Kd =('Kd', 'mean'),
    Bshall_mx =('Bshall', 'max'),
    Bshall_av =('Bshall', 'mean'),
    Bshall_mn =('Bshall', 'min'),
    rel_mx =('reliablefr', 'max'),  # the max of all reliable fractions
    rel_av =('reliablefr', 'mean'),
    rel_mn =('reliablefr', 'min'),
    C_dir_mx=('C_direct', 'max'),  # the max of all clearances  
    C_ind_mx=('C_indir', 'max'), 
    C_unk_mx=('C_unkn', 'max'),
    C_dir_av=('C_direct', 'mean'),  # the mean of all clearances
    C_ind_av=('C_indir', 'mean'), 
    C_unk_av=('C_unkn', 'mean'),
    C_dir_mn=('C_direct', 'min'), # the min of all clearances
    C_ind_mn=('C_indir', 'min'), 
    C_unk_mn=('C_unkn', 'min'),
    BsTID_rel=('Bshall_TID', 'sum')
    )

grd_events['BsTID_rel'] = grd_events['BsTID_rel']/grd_events['count']

grd_events = grd_events.round(decimals=3)

# convert to gpd dataframe
grd_events = gpd.GeoDataFrame(grd_events, geometry='geometry')    
grd_events = grd_events.sort_values(['nickname'])

grd_events.to_file('ci2d3_grd_events_etopo_sd{}.shp'.format(sd))
grd_events=grd_events.loc[:, grd_events.columns != 'geometry']    
grd_events.to_csv('ci2d3_grd_events_etopo_sd{}.csv'.format(sd))

print("\nSummary of run: \n")
print(f"Total grounding events with ETOPO issues: {len(grd_events)}")

print(f"{len(grd)} ice island polygons were analyzed (based on selection criteria)")
print(f"{sum(grd.issueetopo)} ice island polygons ({round(sum(grd.issueetopo)/len(grd)*100,2)} %) had a potential issue with etopo") 



