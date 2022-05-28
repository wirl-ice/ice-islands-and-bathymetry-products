# ice islands and bathymetry products
 Code to compare estimates of ice island keel depths to gridded bathymetry products
 
The main Python script in this repository is a script that determines areas of potenially erroneous bathymetry (too deep) in interpolated gridded bathymetric products.  

It works by assuming a max draft of ice islands and flagging areas where the ice islands are grounded according to the CI2D3 database where they should not be according to the bathymetry.  

The script runs with IBCAOv4 and ETOPO1 data, but could be modifed to work with any gridded product

Note this is a scheme to passively determine where bathymetry might be in error, it is not meant to be definitive in any way as it relies on assumptions. It works like so:

1. Starting from initial thicknesses +/- X sd from Crawford 2018 and Munchow 2014 
   determine the max and min draft 
       note that X is a parameter that can be altered

2. Buffer all the polygons by the worst case in the georef field to ensure that 
   we are looking at any possible pixel under each ice island.  (note >400 m error is removed)

3. Grab all pixels under all ice islands. For IBCAO and ETOPO.  For IBCAO determine 
    if there are any direct measures there

4. At the grounding locations, determine which deepest drafts are above the shallowest bathymetry. 

5. Compare ETOPO1 to IBCAO

6. Compare based on direct vs indirect measurements

7. Export to shapefile for maps


# DATA required:

1. CI2D3 - v1.1 as a shapefile

2. A csv listing all the ice island grounding events that are named (nicknamed) after the last 3  letters of the inst of the first ice island that was observed to be grounding.  A list of all the ice islands in that grounding follows - eg: 
     nickname,lineage
     QQZ,"['20101022_221241_r2_13_QQZ', '20101026_115942_r2_23_OAQ']"

3. IBCAO v4    --  Coverage is North of 64deg only. 
     https://www.gebco.net/data_and_products/gridded_bathymetry_data/arctic_ocean/
     Download the 200x200m grid as geotiff
 Also download: 
     200mx200m Data Type Identifier Grid = This data set identifies the type of data that a grid cell is based on. It follows the TID coding as used for the GEBCO global grid.
     200mx200m Data Source Identifier Grid = This data set identifies the source of data that a grid cell is based on. It follows a mysterious coding that I have yet to learn about (but some data sources like NONNA-100 are apparent)

4. ETOPO1 -- global dataset - 
     https://ngdc.noaa.gov/mgg/global/global.html
     https://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/georeferenced_tiff/

Note for both these datasets, assigned the correct projection to the downloaded file
EPSG 3996 for IBCAO
EPSG 4326 for ETOPO1

