#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci2d3_grd_ibcao_data.py

This script looks at the Type and Source information from the IBCAO data


Created April 2022

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

def dataExplore(tid_img, sid_img, iidf, buffer=1000):
    """
    This function is optional.  It looks at the IBCAO TID (data type identifier)
    and the SID (source type identifier) to explore what data are available near 
    all ice islands.  

    Parameters
    ----------
    tid_img : str
        name of tid tif image file in current directory
    sid_img : str
        name of sid tif image file in current directory
    iidf : geopandas dataframe
        geodataframe with ice islands 
    buffer : numeric, optional
        Define 'near' in metres. The default is 1000.

    Returns
    -------
    None
        but clips tid and sid to the buffered ice island dataset and masks out data that is not of interest
        prints out counts of TIDs and SIDs

    """
    ## Checked here to see what TIDs are near the ice islands -- only did this once (hence it is commented out)
    arr_tid = imgclip(tid_img,'tid_img_clip', iidf.to_crs(ibcao_crs).buffer(1000),maskout=True)
    count_tid = dict(zip(*np.unique(arr_tid, return_counts=True)))
    print("Unique TID values and counts:", count_tid)
    
    ## Looked at the different sources of data.  There is legend to determine what the codes mean
    #https://www.gebco.net/data_and_products/gridded_bathymetry_data/documents/ibcao_v4_data_sets.pdf
    arr_sid = imgclip(sid_img,'sid_img_clip', iidf.to_crs(ibcao_crs).buffer(1000),maskout=True)
    count_sid = dict(zip(*np.unique(arr_sid, return_counts=True)))
    print("Unique SID values and counts:", count_sid)
    
    # find out what SIDs are associated with various TIDs
    count_sid_tid10s = dict(zip(*np.unique(arr_sid[np.where(arr_tid <=20)], return_counts=True)))
    print("Unique SID values and counts for 'TID: direct':", count_sid_tid10s)
    
    count_sid_tid41 = dict(zip(*np.unique(arr_sid[np.where(arr_tid == 41)], return_counts=True)))
    print("Unique SID values and counts for 'TID: indirect 41':", count_sid_tid41)
    
    count_sid_tid42 = dict(zip(*np.unique(arr_sid[np.where(arr_tid == 42)], return_counts=True)))
    print("Unique SID values and counts for 'TID: indirect 42':", count_sid_tid42)
    
    count_sid_tid70 = dict(zip(*np.unique(arr_sid[np.where(arr_tid == 70)], return_counts=True)))
    print("Unique SID values and counts for 'TID: unknown':", count_sid_tid70)
                                  

''' MAIN '''        
# Files and directories
indir = '/home/dmueller/Desktop/bathy'  # directory that contains all input files

ci2d3 = 'CI2D3_v01.1_selected.shp'     # shapefile containing ci2d3 data
lineagefile = 'grounded_lineage.csv'  # a csv with all the names of ice islands that are grounded (more than once)
ibcao_img = "IBCAO_v4_200m.tif"      # the name of the IBCAO data file (geotiff)
tid_img = "IBCAO_v4_200m_TID.tif"    # the name of the IBCAO TID file (geotiff)
sid_img = "IBCAO_v4_200m_SID.tif"    # the name of the IBCAO SID file (geotiff) - optional in practice
ibcao_crs = 'EPSG:3996'   #EPSG code for the IBCAO projection

os.chdir(indir)

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

dataExplore(tid_img, sid_img, ii)  #Run this if you want more info on the ibcao data
