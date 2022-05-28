#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci2d3_bathy.py

Script to compare gridded bathymetric products vs an estimated keel depth for ice islands. 

Created on June 18 26 15:29:58 2021

@author: dmueller
"""

# import libraries
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
import numpy as np


# functions
def thick2draft(thickness, rho_i = 917, rho_w = 1000):
    # calculate ice draft below sea level given the ice thickness
    # assuming hydrostatic equilibrium and tabular profile
  
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
sd = 5   # the number of standard deviations for max draft estimate.

# if you want to specify the max draft (keel depth) for a given calving year, that can be done here
# change these from 0 to whatever value you wish (positive numbers for max draft)
specifymaxdraft2008 = 0
specifymaxdraft2010 = 0
specifymaxdraft2012 = 0

# Files and directories
indir = '/home/dmueller/Desktop/bathy'  # directory that contains all input files
outdir = f'/home/dmueller/Desktop/bathy/sd{sd}'  # directory for script output

ci2d3 = 'CI2D3_v01.1_selected.shp'     # shapefile containing ci2d3 data
lineagefile = 'grounded_lineage.csv'  # a csv with all the names of ice islands that are grounded (more than once)
ibcao_img = "IBCAO_v4_200m.tif"      # the name of the IBCAO data file (geotiff)
tid_img = "IBCAO_v4_200m_TID.tif"    # the name of the IBCAO TID file (geotiff)
sid_img = "IBCAO_v4_200m_SID.tif"    # the name of the IBCAO SID file (geotiff) - optional in practice
etopo1_img = "ETOPO1_Bed_g_geotiff.tif"   # The name of the ETOPO1 file
ibcao_crs = 'EPSG:3996'   #EPSG code for the IBCAO projection
etopo_crs = 'EPSG:4326'   #EPSG code for the ETOPO1 projection

os.chdir(indir)

print(f'Analysis based on a maximum ice draft calculated with {sd} standard deviations from the mean of thickness \n')

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

ii = assignNickname(ii, grdlist, xn=2)

# must reset the index after any selection you do so the concatentation goes well below. 
ii = ii.reset_index(drop=True)
print(f"There are {ii.shape[0]} ice island polygons to examine in a total of {len(ii.nickname.unique())} grounding events\n")

# Thickness and draft
thick2008_mean = 62.3 #m  based on draft calculated with rho_i = 873, rho_w = 1025 (Crawford 2018 JGR)
thick2008_sd = 6.8
# This back-calculates the draft to Humphrey's data
draft2008_max = thick2draft(thick2008_mean+sd*thick2008_sd, rho_i=873, rho_w=1025)
draft2008_min = thick2draft(thick2008_mean-sd*thick2008_sd, rho_i=873, rho_w=1025)

thick2010_mean = 76 #m  based on radar and alitmetry (Munchow et al 2014)
thick2010_sd = 6
# draft for thickest and thinnest possible (3 sigma) - note different densities
draft2010_max = thick2draft(thick2010_mean+sd*thick2010_sd, rho_i=917, rho_w=1025)
draft2010_min = thick2draft(thick2010_mean-sd*thick2010_sd, rho_i=873, rho_w=1026)

thick2012_mean = 182 #m  based on radar and alitmetry (Munchow et al 2014)
thick2012_sd = 16
draft2012_max = thick2draft(thick2012_mean+sd*thick2012_sd, rho_i=917, rho_w=1025)
draft2012_min = thick2draft(thick2012_mean-sd*thick2012_sd, rho_i=873, rho_w=1026)

print('Ice island thickness:')
print(" Thickness in 2008 ranges from : {0:.1f} to {1:.1f} m".format(thick2008_mean-sd*thick2008_sd, thick2008_mean+sd*thick2008_sd) )
print(" Thickness in 2010 ranges from : {0:.1f} to {1:.1f} m".format(thick2010_mean-sd*thick2010_sd, thick2010_mean+sd*thick2012_sd) )
print(" Thickness in 2012 ranges from : {0:.1f} to {1:.1f} m".format(thick2012_mean-sd*thick2012_sd, thick2012_mean+sd*thick2012_sd) )

print('Ice island draft:')
print(" Draft in 2008 ranges from : {0:.1f} to {1:.1f} m".format(draft2008_min, draft2008_max) )
print(" Draft in 2010 ranges from : {0:.1f} to {1:.1f} m".format(draft2010_min, draft2010_max) )
print(" Draft in 2012 ranges from : {0:.1f} to {1:.1f} m".format(draft2012_min, draft2012_max) )
#print("\n NOTE: draft will be recorded as an elevation (like bathymetry) = ice_min = -100 m, ice_max = -50 m")


# Check if the maxdraftYYY inputs are 0 or not.  If they are not zero then 
if specifymaxdraft2008 != 0:
    print('Max draft for 2008 was specified as {0:.1f}'.format(specifymaxdraft2008))
    draft2008_max = specifymaxdraft2008
if specifymaxdraft2010 != 0:
    print('Max draft for 2010 was specified as {0:.1f}'.format(specifymaxdraft2010))
    draft2010_max = specifymaxdraft2010
if specifymaxdraft2012 != 0:
    print('Max draft for 2012 was specified as {0:.1f}'.format(specifymaxdraft2012))
    draft2012_max = specifymaxdraft2012

ii['ice_min'] = -1*draft2008_max
ii.loc[ ii.calvingyr == '2010','ice_min'] = -1*draft2010_max
ii.loc[ii.calvingyr == '2012','ice_min'] = -1*draft2012_max

ii['ice_max'] = -1*draft2008_min
ii.loc[ ii.calvingyr == '2010','ice_max'] = -1*draft2010_min
ii.loc[ii.calvingyr == '2012','ice_max'] = -1*draft2012_min

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
#grd.lat[grd.lat<65]

# this has ibcao and etopo data
grd = grd.loc[grd.lat>64]  
grd = grd.reset_index(drop=True)

# add all these new columns (with bad data flags or default values)

grd.insert(grd.shape[1], "ibcao_max",999.0)  # max value under ice island - used to flag issues - compare w/ ice_min
grd.insert(grd.shape[1], "ibcao_min",999.0)  # not really important... 
grd.insert(grd.shape[1], "ibcao_mean",999.0) # not as important
#See GEBCO TID Grid coding  
# number of pixels by TID
grd.insert(grd.shape[1], "ibcao_pix",-999)  # number of pixels for all
grd.insert(grd.shape[1], "direct_pix",-999)  # TIDs between 10 and 19 - assumed to be very acccuate
grd.insert(grd.shape[1], "indir_pix",-999)  # TIDs in 40s  - assumed to be interpolations
grd.insert(grd.shape[1], "unkn_pix",-999)   # all unknowns is TID 70 which corresponds to NONNA data
grd.insert(grd.shape[1], "reliablefr",-999.0)# fraction of all pixels that are direct+unknown 
### clearance 1  is ibcao_max - the ice_min where TID is direct, indirect, unknown
grd.insert(grd.shape[1], "direct_cl1",-999.0)  # TIDs between 10 and 19 - assumed to be very acccuate
grd.insert(grd.shape[1], "indir_cl1",-999.0)  # TIDs in 40s  - assumed to be interpolations
grd.insert(grd.shape[1], "unkn_cl1",-999.0)   # all unknowns is TID 70 which corresponds to NONNA data

### clearance 2  is ibcao_min - the ice_min where TID is direct, indirect, unknown
grd.insert(grd.shape[1], "direct_cl2",-999.0)  # TIDs between 10 and 19 - assumed to be very acccuate
grd.insert(grd.shape[1], "indir_cl2",-999.0)  # TIDs in 40s  - assumed to be interpolations
grd.insert(grd.shape[1], "unkn_cl2",-999.0)   # all unknowns is TID 70 which corresponds to NONNA data

### More info abotu the high pixels 
grd.insert(grd.shape[1], "highPt_TID",999)  # TID of the highest point 
grd.insert(grd.shape[1], "hi_ind_pix",999)  # Number of indirect pixels above highest reliable (direct/unknown)
## Etopo data
grd.insert(grd.shape[1], "etopo_max",999.0)
grd.insert(grd.shape[1], "etopo_min",999.0)
grd.insert(grd.shape[1], "etopo_mean",999.0)
grd.insert(grd.shape[1], "etopo_pix",-999)
## Issue flags
grd.insert(grd.shape[1], "issueibcao",0)
grd.insert(grd.shape[1], "issueetopo",0)

os.chdir(outdir)

for i in range(len(grd)):
    # go through each row in the dataframe
    iceisl = grd.iloc[[i]]
    
    # save images and extract numpy array of the BUFFERED ice island 
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
        
    grd.iloc[i,grd.columns.get_loc('ibcao_max')] = np.nanmax(b)
    grd.iloc[i,grd.columns.get_loc('ibcao_min')] = np.nanmin(b)
    grd.iloc[i,grd.columns.get_loc('ibcao_mean')] = np.nanmean(b)
    
    grd.iloc[i,grd.columns.get_loc('ibcao_pix')] = b[np.logical_not(np.isnan(b))].size
    grd.iloc[i,grd.columns.get_loc('direct_pix')]  = counts[vals<20].sum()   # NOTE ASSUMING THERE ARE NO TIDs of 0
    grd.iloc[i,grd.columns.get_loc('indir_pix')] = counts[vals<50].sum()-counts[vals<20].sum()
    grd.iloc[i,grd.columns.get_loc('unkn_pix')]  = counts[vals>50].sum()
    grd.iloc[i,grd.columns.get_loc('reliablefr')] = ((counts[vals>50].sum()+counts[vals<20].sum())/b[np.logical_not(np.isnan(b))].size)

    grd.iloc[i,grd.columns.get_loc('direct_cl1')]  = np.nan if np.where(t<20)[0].size ==0 else iceisl.ice_min - b[np.where(t<20)].max()
    grd.iloc[i,grd.columns.get_loc('indir_cl1')] = np.nan if np.where((t>20) & (t<70))[0].size ==0 else iceisl.ice_min - b[np.where((t>20) & (t<70))].max() 
    grd.iloc[i,grd.columns.get_loc('unkn_cl1')]  = np.nan if np.where(t>50)[0].size ==0 else iceisl.ice_min - b[np.where(t>50)].max()

    grd.iloc[i,grd.columns.get_loc('direct_cl2')]  = np.nan if np.where(t<20)[0].size ==0 else iceisl.ice_min - b[np.where(t<20)].min()
    grd.iloc[i,grd.columns.get_loc('indir_cl2')] = np.nan if np.where((t>20) & (t<70))[0].size ==0 else iceisl.ice_min - b[np.where((t>20) & (t<70))].min()
    grd.iloc[i,grd.columns.get_loc('unkn_cl2')]  = np.nan if np.where(t>50)[0].size ==0 else iceisl.ice_min - b[np.where(t>50)].min()

    # TID of the highest point
    grd.iloc[i,grd.columns.get_loc('highPt_TID')] = t[np.where(b==np.nanmax(b))].min()  # in case of multiple returns take min
    # Number of indirect pixels above highest reliable (direct/unknown)
    try: 
        # There are numerous issues that could go wrong - no indirect for example
        b_ind = b[np.where((t>20) & (t<70))]   #bathy from indirect TID only
        #Highest reliable: 
        rel_max = b_ind.min() if np.where((t>50) | (t<20)) else b[np.where((t>50) | (t<20))].max()
        hi_ind_pix = np.where(b_ind>rel_max)[0].size
    except: 
        hi_ind_pix = np.nan
    grd.iloc[i,grd.columns.get_loc('hi_ind_pix')]  = hi_ind_pix
    
    # Note that sometimes you get an 'all nan' slice so this means the etopo is mucked up    
    grd.iloc[i,grd.columns.get_loc('etopo_max')] = np.nanmax(e)
    grd.iloc[i,grd.columns.get_loc('etopo_min')] = np.nanmin(e)
    grd.iloc[i,grd.columns.get_loc('etopo_mean')] = np.nanmean(e)
    grd.iloc[i,grd.columns.get_loc('etopo_pix')] = e[np.logical_not(np.isnan(e))].size
    

    # if this is true, we have an issue with ibcao bathy data
    if iceisl.ice_min.values[0] > np.nanmax(b):
        grd.iloc[i,grd.columns.get_loc('issueibcao')] = 1
    if iceisl.ice_min.values[0] > np.nanmax(e):
        grd.iloc[i,grd.columns.get_loc('issueetopo')] = 1
        
    # no sense keeping these images if there are no issues    
    if grd.iloc[i,grd.columns.get_loc('issueibcao')] == 0: 
        os.remove('{}_ibcao_b.tif'.format(iceisl.inst.values[0]))
        os.remove('{}_ibcao_t.tif'.format(iceisl.inst.values[0]))
    if grd.iloc[i,grd.columns.get_loc('issueetopo')] == 0:
        os.remove('{}_etopo_b.tif'.format(iceisl.inst.values[0]))  

# export data table - one row for each grounded ice island 
# rounding for legibility and ONLY exporting the IBCAO issues 
grd1 = grd.round(decimals=4)
grd1 = grd1.loc[grd['issueibcao'] == 1]
grd1 = grd1.sort_values(['nickname','inst'])
grd1.to_file(f'ci2d3_grd_ii_sd{sd}.shp')
grd2=grd.loc[:, grd1.columns != 'geometry']
grd2.to_csv(f'ci2d3_grd_ii_sd{sd}.csv')

### export summary table  
# group and count by issue
grdgrp = grd.groupby(['nickname', 'issueibcao'])['nickname'].agg('count')
# move index to column
grdgrp = grdgrp.reset_index(level=[1])
# only want groups with issues
grdgrp = grdgrp.loc[grdgrp.issueibcao == 1]

# select from the detailed dataframe - uncomment next line to get all groundings that have ibcao issues
# but also the ice islands that don't have any issue in those groundings
grd_summary = grd[grd['nickname'].isin(grdgrp.index)]

# This line of code will only select the ibcao issues,  comment to leave in groundings that have issues and not in
grd_summary = grd_summary.loc[grd_summary.issueibcao == 1]

# need to get stats for all events
grd_events = grd_summary.groupby(['nickname', 'issueibcao']).agg(
    first_inst =('inst', 'first'),
    geometry =('geometry', 'first'),
    ice_min =('ice_min', 'mean'),
    bmax_mean =('ibcao_max', 'mean'),
    bmax_min =('ibcao_max', 'min'),
    bmax_max =('ibcao_max', 'max'),
    rel_mean =('reliablefr', 'mean'),
    rel_min =('reliablefr', 'min'),
    rel_max =('reliablefr', 'max'),
    cl1_dir_mx=('direct_cl1', 'max'),  # the max of all min clearances  
    cl1_ind_mx=('indir_cl1', 'max'), 
    cl1_unk_mx=('unkn_cl1', 'max'),
    cl1_dir_av=('direct_cl1', 'mean'),  # the mean of all min clearances
    cl1_ind_av=('indir_cl1', 'mean'), 
    cl1_unk_av=('unkn_cl1', 'mean'),
    cl1_dir_mn=('direct_cl1', 'min'), # the min of all min clearances
    cl1_ind_mn=('indir_cl1', 'min'), 
    cl1_unk_mn=('unkn_cl1', 'min')
    )

grd_events = grd_events.round(decimals=3)

# convert to gpd dataframe
grd_events = gpd.GeoDataFrame(grd_events, geometry='geometry')    
grd_events = grd_events.sort_values(['nickname'])

grd_events.to_file('ci2d3_grd_events_sd{}.shp'.format(sd))
grd_events=grd_events.loc[:, grd_events.columns != 'geometry']    
grd_events.to_csv('ci2d3_grd_events_sd{}.csv'.format(sd))


print(f"\nSummary of run (std of {sd}) \n")

print(f"Total grounding events with IBCAO issues: {len(grd_events)}")

print(f"{len(grd)} ice islands were analyzed (based on selection criteria)")
print(f"{sum(grd.issueibcao)} ice islands ({round(sum(grd.issueibcao)/len(grd)*100,2)} %) had a potential issue with ibcao") 
#print(f"{sum(grd.issueetopo)} ice islands ({round(sum(grd.issueetopo)/len(grd)*100,2)} %) had a potential issue with etopo") 
#print(f"{len(grd.loc[(grd['issueetopo'] == 1) & (grd['issueibcao'] == 1)])} ice islands had a potential issue with both ibcao and etopo")

print(f"The minimum of all event minimum clearances for TID direct is {grd_events.cl1_dir_mn.min()} m")
print(f"The minimum of all event minimum clearances for TID indirect is {grd_events.cl1_ind_mn.min()} m")
print(f"The minimum of all event minimum clearances for TID unknown is {grd_events.cl1_unk_mn.min()} m")

#TODO: 

#There are 2 ice islands that are unnamed, check    

#Check output bmax_max seems to wander

# get rid of FutureWarning from output

# format output table Something like:
#Lat, lon (of first obs?), issue with IBCAO and/or ETOPO, max percentage of direct pixels if IBCAO flagged
#lat, lon, first grounding inst, last grounding inst?, 5sd ice_min, bmax_max, rel_max, min(cl1_dir_mn, cl1_unk_mn), cl1_ind_min. 

#Document table / output fields in README
