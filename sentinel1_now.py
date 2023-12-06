# get sentinel1 file closest to labeled image date
# how to deal with compose the full AOI image cover by compositing two adjacent scenes in this case
# deleted 48 because images need to be composed
# https://developers.google.com/earth-engine/tutorials/community/sar-basics#sentinel-1_coverage

# %%
import ee
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
from os import listdir
import os
from datetime import datetime

# Trigger the authentication flow.# Initialize the library.
ee.Authenticate()
ee.Initialize()
# %%
# get 1kmx1km chunk of sentinel1 centered on location for JJA for 2014-2023

location1_lat, location1_lon = 40.936639,-106.339194
location2_lat, location2_lon = 38.258194,-107.546111

# %%
# Define the point of interest (longitude, latitude)
point_of_interest = ee.Geometry.Point(location2_lon, location2_lat)  # Replace with your AOI coordinates



# Define the point of interest (longitude, latitude)
#point_of_interest = ee.Geometry.Point(-105.2705, 40.0150)  # Replace with your AOI coordinates

# Define the time range
#years = [2021, 2022]
years = range(2017,2024)
months = ['06', '07', '08']

# Loop through years and months to fetch data and export to Google Drive
for year in years:
    for month in months:
        # Define start and end dates for each month
        start_date = ee.Date.fromYMD(year, int(month), 1)
        end_date = start_date.advance(1, 'month').advance(-1, 'day')
        
        # Define a 1 km square around the point of interest
        aoi = point_of_interest.buffer(500).bounds()
        
        # Filter Sentinel-1 GRD data
        sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select(['VV', 'VH'])  # Select VV and VH polarization bands
        
        # Get a list of all images in the collection
        image_list = sentinel1.toList(sentinel1.size())
        
        # Loop through each image and export to Google Drive
        for i in range(image_list.size().getInfo()):
            image = ee.Image(image_list.get(i))
            
            # Get image acquisition date and time
            date_time = ee.Date(image.get('system:time_start')).format("YYYY-MM-dd_HH-mm-ss").getInfo()
            
            # Export the image to Google Drive with date and time in filename
            task = ee.batch.Export.image.toDrive(image=image,
                                                 description=f'loc2_{date_time}',
                                                 folder='GEE_Export',
                                                 scale=10,
                                                 region=aoi,
                                                 fileFormat='GeoTIFF',
                                                 formatOptions={'cloudOptimized': True},
                                                 crs='EPSG:4326')
            
            # Start the export task
            task.start()



# %%
