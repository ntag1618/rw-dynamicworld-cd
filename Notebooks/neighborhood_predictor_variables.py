#Import necessary libraries
import os
import ee
import numpy as np
import sklearn
import statsmodels.api as sm
import pandas as pd
import random
import json
import time
num_seed=30
random.seed(num_seed)



#Image bands must be ordered by increasing years
def get_year_stack_image_collection(image, band_names, band_indices=[-1,0,1]):
    '''
    Function returns image collection of images where each band is taken from the band_indices. If inputted bands do
                do not follow the band indices, that image will not be returned. 
                For example if one band index is less than 0, an image for the first band will not be returned
                because there is not a band corresponding to that index.
    Inputs:
        image: image where each band represents the land cover classification for a year, bands ordered by 
                increasing years
        band_names: list of band names in the image
        band_indices: list of indices you want to collect from the image, the default [-1,0,1] will return an 
                image collection where each image will have the bands [previous year, current year, following year]
    Returns:
        out_image_list: an image collection where each image corresponds to a band in band_names, where the bands 
                of the image correspond to the band_indices input
                
    Example:
        Inputs:
            image = image of land cover classification for years [1986,1987,1988,1989]
            band_names = [1986,1987,1988,1989]
            band_indices = [-1,0,1]
        Returns:
            out_image_list = image collection with the following images:
                image 1: bands: [1986,1987,1988], property {'OriginalBand': 1987}
                image 2: bands: [1987,1988,1989], property {'OriginalBand': 1988}
            (an image for 1986 is not included because there is not a year before 1986,
             and an image for 1989 is not included because there is not a year after 1989)
    '''
    out_image_list = []
    for i,band_name in enumerate(band_names):
        #indices = i_
        if all(np.array([int(i+x) for x in band_indices])>=0):
            try:
                band_list = [band_names[i+x] for x in band_indices]
                out_image = ee.Image.cat(image.select(band_list))
                out_image = out_image.set(ee.Dictionary({'OriginalBand':band_name}))
                out_image_list.append(out_image)
            except:
                None
    
    return ee.ImageCollection(out_image_list)


#Functions for binary land cover change properties
def lc_one_change(image):
    '''
    Determines if there was one change occurance from year i to year i+1. Returns an image with values:
    1 if state(i) != state(i+1)
    0 if state(i) == state(i+1)
    '''
    band_names = image.bandNames()
    out_image = image.select([band_names.get(0)]).neq(image.select([band_names.get(1)]))
    out_image = out_image.select(out_image.bandNames(),[band_names.get(0)])
    out_image = out_image.set(ee.Dictionary({'OriginalBand':band_names.get(0)}))
    return out_image

def lc_no_change(image):
    '''
    Determines if there was no change occurance from year i to year i+1. Returns an image with values:
    1 if state(i) != state(i+1)
    0 if state(i) == state(i+1)
    '''
    band_names = image.bandNames()
    out_image = image.select([band_names.get(0)]).eq(image.select([band_names.get(1)]))
    out_image = out_image.select(out_image.bandNames(),[band_names.get(0)])
    out_image = out_image.set(ee.Dictionary({'OriginalBand':band_names.get(0)}))
    return out_image

def lc_reverse(image):
    '''
    Determines if change that occured from i to i+1 reversed back to state i in i+2
    1 if state(i) != state(i+1) and state(i) == state(i+2)
    0 otherwise
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    next_next_year = image.select([band_names.get(2)])
    
    returnback = current_year.eq(next_next_year)
    changed = current_year.neq(next_year)
    out_image = returnback.bitwise_and(changed)
    out_image = out_image.select(out_image.bandNames(),[band_names.get(0)])
    out_image = out_image.set(ee.Dictionary({'OriginalBand':band_names.get(0)}))
    return out_image

def lc_change_to_another(image):
    '''
    Determines if change occured from i to i+1 and change occured in i+1 to i+2 where state(i)!=state(i+2)
    1 if state(i) != state(i+1) and state(i) != state(i+2) and state(i+1) != state(i+2)
    0 otherwise
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    next_next_year = image.select([band_names.get(2)])
    
    changed = current_year.neq(next_year)
    changed_again = next_year.neq(next_next_year)
    not_reversed = current_year.neq(next_next_year)
    
    out_image = changed.bitwise_and(changed_again.bitwise_and(not_reversed))
    out_image = out_image.select(out_image.bandNames(),[band_names.get(0)])
    out_image = out_image.set(ee.Dictionary({'OriginalBand':band_names.get(0)}))
    return out_image

def lc_consistent_change_one_year(image):
    '''
    Determines if change that occured from i to i+1 stayed in i+2
    1 if state(i) != state(i+1) and state(i+1) == state(i+2)
    0 otherwise
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    next_next_year = image.select([band_names.get(2)])
    
    changed = current_year.neq(next_year)
    stayed = next_year.eq(next_next_year)
    
    out_image = changed.bitwise_and(stayed)
    out_image = out_image.select(out_image.bandNames(),[band_names.get(0)])
    out_image = out_image.set(ee.Dictionary({'OriginalBand':band_names.get(0)}))
    return out_image

def lc_consistent_change_two_years(image):
    '''
    Determines if change that occured from i to i+1 stayed in i+2 and i+3
    1 if state(i) != state(i+1) and state(i+1) == state(i+2) and state(i+1) == state(i+3)
    0 otherwise
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    next_next_year = image.select([band_names.get(2)])
    next_next_next_year = image.select([band_names.get(3)])
    
    changed = current_year.neq(next_year)
    stayed = next_year.eq(next_next_year)
    stayed_again = next_year.eq(next_next_next_year)
    
    out_image = changed.bitwise_and(stayed.bitwise_and(stayed_again))
    out_image = out_image.select(out_image.bandNames(),[band_names.get(0)])
    out_image = out_image.set(ee.Dictionary({'OriginalBand':band_names.get(0)}))
    return out_image

def lc_year_after(image):
    '''
    Returns land cover class for following year
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    out_image = next_year.select(next_year.bandNames(),[band_names.get(0)])
    out_image = out_image.set(ee.Dictionary({'OriginalBand':band_names.get(0)}))
    return out_image

#Functions I've written to try to do this sampling

#Function to convert feature collection to pandas dataframe
def get_dataframe_from_feature_collection(feature_collection, property_names):
    df = pd.DataFrame()
    for property_name in property_names:
        property_values = feature_collection.aggregate_array(property_name).getInfo()
        df[property_name] = property_values
    return df

#Function to convert pandas dataframe to feature collection
def convert_points_df_to_feature_collection(df,projection='EPSG:4326',lat_name='latitude',lon_name='longitude'):
    feature_collection_list = []
    for i,row in df.iterrows():
        geometry = ee.Geometry.Point([row[lon_name],row[lat_name]])#,projection)
        row_dict = row.to_dict()
        row_feature = ee.Feature(geometry,row_dict)
        feature_collection_list.append(row_feature)
    return ee.FeatureCollection(feature_collection_list)

#Function to convert pandas dataframe to feature collection
def convert_point_df_to_feature(series,projection='EPSG:4326',lat_name='latitude',lon_name='longitude'):
    geometry = ee.Geometry.Point([series[lon_name],series[lat_name]])#,projection)
    row_dict = series.to_dict()
    row_feature = ee.Feature(geometry,row_dict)
    return row_feature

#Function to sample image data at point locations (sampleBandPoints) and rename new property to image_name
def getSampleImageData(image, sampleBandPoints, image_name):
    #Sample image data at point locations
    #reduceRegions(collection, reducer, scale, crs, crsTransform, tileScale)
    sampleImageData = image.reduceRegions(
        collection=sampleBandPoints,
        reducer=ee.Reducer.first(),
        crs=crs,
        crsTransform=crsTransform
        )
    #Rename sampled values from "first" to image_name
    sampleImageData = sampleImageData.map(lambda x: x.set({image_name:x.get('first')}))
    return sampleImageData
