# Ice Islands and Gridded Bathymetry Products
 Code to compare estimates of ice island keel depths to gridded bathymetry products
 
The main Python script in this repository is a script that determines areas of potenially erroneous bathymetry (too deep) in interpolated gridded bathymetric products.  

It works by assuming a maximum possible draft (Kd) of ice islands and flagging areas where the ice islands are grounded according to the CI2D3 database where they should not be according to the bathymetry.  

The script runs with IBCAOv4 (mainly) and ETOPO1 data (as an add on), but could be modifed to work with any gridded bathymetry product

Note this is a scheme to passively determine where gridded bathymetry might be inaccurate, it is not meant to be definitive in any way as it relies on assumptions. It works like so:

1. Starting from initial thicknesses conservatively determine the Kd which is the maximum (aka the deepest) draft that all ice islands from each calving event should have
	This can work by specifying a mean thickness and adding a given number of standard deviations OR by specifying the maximum thickness directly
        
2. Buffer all the polygons by the worst case in the georef field to ensure that 
   we are looking at any possible pixel under each ice island.  (note polygons with >400 m error are removed)

3. Grab all bathymetry pixels under all ice islands using IBCAO (and ETOPO as well).  For IBCAO using the data type info determine 
    if there are any direct or unknown measures there (since these are likely reliable)

4. At the grounding locations, determine which Kd are above the shallowest bathymetry (Bshall and eBshall). 

5. Compare based on direct/unknown (reliable bathymetry) vs indirect measurements (IBCAO only)

6. Make summary tables 

7. Export to shapefile for mapping or to CSV files for futher analysis


# DATA required:

1. CI2D3 - v1.1 as a shapefile 

2. A csv listing all the ice island grounding events that are named (nicknamed) after the last 3 letters of the inst of the first ice island that was observed to be grounding.  A list of all the ice islands in that grounding follows - eg: 
     nickname,lineage
     QQZ,"['20101022_221241_r2_13_QQZ', '20101026_115942_r2_23_OAQ']"
     
     Note this file is in the github repo (along with a script (grounded_lineage.py) to generate it from grounded_lineage_2010.csv and grounded_lineage_2012.csv (which are not ideal formats)
     
3. IBCAO v4    --  Coverage is North of 64deg only. 
     https://www.gebco.net/data_and_products/gridded_bathymetry_data/arctic_ocean/
     Download the 200x200m grid as geotiff
 Also download: 
     200mx200m Data Type Identifier Grid = This data set identifies the type of data that a grid cell is based on. It follows the TID coding as used for the GEBCO global grid.
     200mx200m Data Source Identifier Grid = This data set identifies the source of data that a grid cell is based on. It follows a mysterious coding that I have yet to learn about (but some data sources like NONNA-100 are apparent)

4. ETOPO1 -- global dataset - 
     https://ngdc.noaa.gov/mgg/global/global.html
     https://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/georeferenced_tiff/

Note for both these datasets, the correct projection to the downloaded file must be assigned (check to make sure)
EPSG 3996 for IBCAO
EPSG 4326 for ETOPO1

# Software Dependencies

To run this script you will need the following libraries installed in your Python environment. 
* geopandas
* rasterio
* numpy

These can be added using conda like so:
 conda install geopandas rasterio numpy
 
 
# Script outputs

The script will output some information to the screen (which can be captured to a text file) and 2 sets of files in 2 formats (shapefile and comma-separated-values): 

Units are metres for and the datum is water level for most fields

## First Output Table

The first is a data file [ci2d3_grd_ii_sd{sd}] which contains the following fields for each ice island polygon that is grounded and has a clearance that is positive in the bathymetry data. 
[NOTE this means that ice islands with potential ETOPO and IBCAO issues are listed] 

The first columns are from the CI2D3 database.  See that documentation for explanations: 

* inst,lineage,calvingyr,calvingloc,area,perimeter,length,lon,lat,scenedate,imgref,mothercert,shpcert,georef,ddinfo,sensor,beam_mode,pol

The next serveral columns are: 

* nickname - the short 3 letter identifier for the polygon that at the beginning of the grounding event
* firstgrnd - T/F is this the first time this ice island is grounded in this location?
* Kd - the deepest draft expected based on estimate of thickness

The following columns refer to IBCAO only
* Bshall - shallowest bathymetry under ice island - used to flag issues - compare w/ Kd to get Clearance [B is for Bathymetry]
* Bdeep - deepest bathymetry under ice island (not super important) 
* Bmean - mean bathymetry under ice island (not as important)
* P_n - number of Pixels found under the ice island  [P is for Pixels]
* P_direct - TIDs between 10 and 19 - assumed to be very acccuate
* P_indir - TIDs in 40s  - assumed to be interpolations
* P_unkn - all unknowns is TID 70, which corresponds to NONNA data (and there may be other data as part of that - Amundsen for example)
* reliablefr - fraction of all pixels that are direct+unknown 
* C  - Smallest Clearance (Bshall - Kd) for all data types - When clearance is positive this indicates a problem with bathymetric data.  Negative clearance is left blank [C is for Clearance]
* C_direct - Smallest clearance between Kd and the shallowest bathymetry where TID is a direct measure (reliable)
* C_indir - Smallest clearance between Kd and the shallowest bathymetry where TID is an indirect measure (unreliable)
* C_unkn - Smallest clearance between Kd and the shallowest bathymetry where TID is an unknown measure (reliable)
* Bshall_TID - The TID of the shallowest pixel (under 20 is direct; 40 and 41 is indirect; 70 and 71 is unknown)
* hi_ind_pix - Number of indirect pixels above highest reliable (direct & unknown) data types

The following columns refer to ETOPO only
* eBshall - shallowest bathymetry under ice island
* eBdeep - deepest bathymetry under ice island
* eBmean - mean bathymetry under ice island
* eP_n  - number of pixels found under the ice island
* eC - Smallest clearance (eBshall - Kd); see above for definition

These are flags (1 or 0) for issues with either IBCAO or ETOPO (note the current code exports only rows with one or both == 1)
* issueibcao - 1 = a positive clearance value between Kd and Bshall (Bathymetry from IBCAO)
* issueetopo- 1 = a positive clearance value between Kd and eBshall (Bathymetry from ETOPO) 

## Second Output Table  (version IBCAO)
The second is a summary table that aggregates the above table by grounding event.  
 Note that this only outputs events based on the IBCAO issues

* nickname - the nickname of the first grounded polygon to uniquely identify the grounding event
* count - the number of grounded ice islands that have a positive clearance value in the grounding event
* first_inst - the full CI2D3 identifier of the first grounded polygon in the grounding event with a positive clearance
* lat - the latitude of the 'first_inst' polygon described above 
* lon - the longitude of the 'first_inst' polygon described above
* Kd - the deepest draft expected based on estimate of thickness from the calving event
* Bshall_av - mean of the shallowest bathymetry under ice island for all polygons that were in this grounding event
* Bshall_mn - minimum of the shallowest bathymetry under ice island for all polygons that in this grounding event
* Bshall_mx - maximum of the shallowest bathymetry under ice island for all polygons that in this grounding event
* rel_av - mean of the reliable fraction across all polygons in this grounding event
* rel_mn - minimum of the reliable fraction across all polygons in this grounding event
* rel_mx - maximum of the reliable fraction across all polygons in this grounding event

* C_dir_mx - maximum of C_direct across all polygons in this grounding event
* C_ind_mx - maximum of C_indir across all polygons in this grounding event
* C_unk_mx  - maximum of C_unkn across all polygons in this grounding event 
* C_dir_av - mean of C_direct across all polygons in this grounding event
* C_ind_av -  mean of C_indir across all polygons in this grounding event
* C_unk_av -  mean of C_unkn across all polygons in this grounding event
* C_dir_mn -  minimum of C_direct across all polygons in this grounding event
* C_ind_mn -  minimum of C_indir across all polygons in this grounding event
* C_unk_mn -  minimum of C_unkn across all polygons in this grounding event
* BsTID_rel - The fraction of polygons in this grounding event that have a reliable Bshall_TID (under 20 or 70 or above)

## Second Output Table (version ETOPO)
The second is a summary table that aggregates the above table by grounding event.  
 Note that this only outputs events based on the ETOPO issues

* This table has the same columns as the one above to Bshall_mx; the remainder have no equivalent in ETOPO and are therefore not included. 


## Other outputs

* In addition to these data files, the script saves the bathymetry products as a raster geotiff image clipped to each ice island polygon that has issueibcao and issueetopo == 1. 
* There are some print statements that may be useful.  Capture these by piping output to a text file:  >python ci2d3_grd_bathy.py > sd6_run.txt


