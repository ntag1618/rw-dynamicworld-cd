#Import necessary libraries
import os
import ee
import numpy as np
import sklearn
import statsmodels.api as sm
import pandas as pd
import random
import json
import calendar
import time


#Image bands must be ordered by increasing years
def getYearStackIC(image, band_names, band_indices=[-1,0,1]):
    """
    Function takes an image with bands for each time period (e.g. annual) and returns a restacked image collection where for each image,
                the bands are taken from band_indices of the inputted image. This function can be used if you want to get an image collection 
                where each image has bands in a certain order, such as the year before, the central year, and the year after.
                If the band indices cannot be selected from the input image, that image will not be returned. 
                For example if one band index is less than 0, an image for the first band will not be returned
                because there is not a band corresponding to that index.
    Args:
        image (ee.Image): image where each band represents the land cover classification for a time period (e.g. year)
        band_names (List of strings): list of band names in the image, bands names should be ordered by 
                increasing years
        band_indices (List or np.array of integers): list of indices you want to collect from the image, the default [-1,0,1] will return an 
                image collection where each image will have the bands [previous year, current year, following year]
    Returns:
        out_image_list: an ee.ImageCollection where each image corresponds to a band in band_names, where the bands 
                of the image correspond to the band_indices input
                
    Example:
        Args:
            image = image of land cover classification for years [1986,1987,1988,1989]
            band_names = [1986,1987,1988,1989]
            band_indices = [-1,0,1]
        Returns:
            out_image_list = image collection with the following images:
                image 1: bands: [1986,1987,1988], property {'OriginalBand': 1987}
                image 2: bands: [1987,1988,1989], property {'OriginalBand': 1988}
            (an image for 1986 is not included because there is not a year before 1986,
             and an image for 1989 is not included because there is not a year after 1989)
    """
    out_image_list = []
    for i,band_name in enumerate(band_names):
        if all(np.array([int(i+x) for x in band_indices])>=0):
            try:
                band_list = [band_names[i+x] for x in band_indices]
                out_image = ee.Image.cat(image.select(band_list))
                out_image = out_image.set(ee.Dictionary({'OriginalBand':band_name}))
                out_image_list.append(out_image)
            except:
                None
                #print('Inputted band indices do not match inputted image for band {}'.format(band_name))
    
    return ee.ImageCollection(out_image_list)


#Functions for binary land cover change properties
def LC_OneChange(image):
    '''
    Function to determine if there was one change occurance from year i to year i+1. Returns an image with values:
    1 if state(i) != state(i+1)
    0 if state(i) == state(i+1)
    
    Compatible with outputs from getYearStackIC, the image must have a property "OriginalBand" determining the central year
    
    Args:
        image (ee.Image): and image with 2 bands, state(i) and state(i+1)
    Returns:
        And ee.Image defined above
    '''
    band_names = image.bandNames()
    out_image = image.select([band_names.get(0)]).neq(image.select([band_names.get(1)]))
    out_image = out_image.rename([image.get('OriginalBand')])
    out_image = out_image.set('OriginalBand',image.get('OriginalBand'))
    return out_image

def LC_NoChange(image):
    '''
    Function to determine if there was no change occurance from year i to year i+1. Returns an image with values:
    1 if state(i) != state(i+1)
    0 if state(i) == state(i+1)
    
    Compatible with outputs from getYearStackIC, the image must have a property "OriginalBand" determining the central year
    
    Args:
        image (ee.Image): and image with 2 bands, state(i) and state(i+1)
    Returns:
        And ee.Image defined above
    '''
    band_names = image.bandNames()
    out_image = image.select([band_names.get(0)]).eq(image.select([band_names.get(1)]))
    out_image = out_image.rename([image.get('OriginalBand')])
    out_image = out_image.set('OriginalBand',image.get('OriginalBand'))
    return out_image

def LC_Reverse(image):
    '''
    Function to determine if change that occured from i to i+1 reversed back to state i in i+2
    1 if state(i) != state(i+1) and state(i) == state(i+2)
    0 otherwise
    
    Compatible with outputs from getYearStackIC, the image must have a property "OriginalBand" determining the central year
    
    Args:
        image (ee.Image): and image with 3 bands, state(i), state(i+1), and state(i+2)
    Returns:
        And ee.Image defined above
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    next_next_year = image.select([band_names.get(2)])
    
    returnback = current_year.eq(next_next_year)
    changed = current_year.neq(next_year)
    out_image = returnback.bitwise_and(changed)
    out_image = out_image.rename([image.get('OriginalBand')])
    out_image = out_image.set('OriginalBand',image.get('OriginalBand'))
    return out_image

def LC_ChangeToAnother(image):
    '''
    Function to determine if change occured from i to i+1 and change occured in i+1 to i+2 where state(i)!=state(i+2)
    1 if state(i) != state(i+1) and state(i) != state(i+2) and state(i+1) != state(i+2)
    0 otherwise
    
    Compatible with outputs from getYearStackIC, the image must have a property "OriginalBand" determining the central year
    
    Args:
        image (ee.Image): and image with 3 bands, state(i), state(i+1), and state(i+2)
    Returns:
        And ee.Image defined above
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    next_next_year = image.select([band_names.get(2)])
    
    changed = current_year.neq(next_year)
    changed_again = next_year.neq(next_next_year)
    not_reversed = current_year.neq(next_next_year)
    
    out_image = changed.bitwise_and(changed_again.bitwise_and(not_reversed))
    out_image = out_image.rename([image.get('OriginalBand')])
    out_image = out_image.set('OriginalBand',image.get('OriginalBand'))
    return out_image

def LC_ConsistentChangeOneYear(image):
    '''
    Function to determine if change that occured from i to i+1 stayed in i+2
    1 if state(i) != state(i+1) and state(i+1) == state(i+2)
    0 otherwise
    
    Compatible with outputs from getYearStackIC, the image must have a property "OriginalBand" determining the central year
    
    Args:
        image (ee.Image): and image with 3 bands, state(i), state(i+1), and state(i+2)
    Returns:
        And ee.Image defined above
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    next_next_year = image.select([band_names.get(2)])
    
    changed = current_year.neq(next_year)
    stayed = next_year.eq(next_next_year)
    
    out_image = changed.bitwise_and(stayed)
    out_image = out_image.rename([image.get('OriginalBand')])
    out_image = out_image.set('OriginalBand',image.get('OriginalBand'))
    return out_image

def LC_ConsistentChangeTwoYears(image):
    '''
    Function to determine if change that occured from i to i+1 stayed in i+2 and i+3
    1 if state(i) != state(i+1) and state(i+1) == state(i+2) and state(i+1) == state(i+3)
    0 otherwise
    
    Compatible with outputs from getYearStackIC, the image must have a property "OriginalBand" determining the central year
    
    Args:
        image (ee.Image): and image with 4 bands, state(i), state(i+1), state(i+2), and state(i+3)
    Returns:
        And ee.Image defined above
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
    out_image = out_image.rename([image.get('OriginalBand')])
    out_image = out_image.set('OriginalBand',image.get('OriginalBand'))
    return out_image

def LC_YearAfter(image):
    '''
    Function to returns land cover class for following year
    Compatible with outputs from getYearStackIC, the image must have a property "OriginalBand" determining the central year
    
    Args:
        image (ee.Image): and image with 2 bands, state(i) and state(i+1)
    Returns:
        And ee.Image defined above
    '''
    band_names = image.bandNames()
    current_year = image.select([band_names.get(0)])
    next_year = image.select([band_names.get(1)])
    out_image = next_year.rename([image.get('OriginalBand')])
    out_image = out_image.set('OriginalBand',image.get('OriginalBand'))
    return out_image


def getTemporalProbabilityDifference(probability_collection, date_1_start, date_1_end, date_2_start, date_2_end, reduce_method='median'):
    """
    Function to calculate the difference in land cover probabilities reduced across two time periods. The first date is defined as 
        date_1_start through date_1_end. The second date is defined as date_2_start through date_2_end. The probabilities are reduced
        based on the reduce_method input.

    Args:
        probability_collection (ee.ImageCollection): an ImageCollection of images of classified land cover probabilities, each image must have
                                either the start_date and end_date properties defined in order to filter by season definitions
        date_1_start (ee.Date): start date of the first period
        date_1_end (ee.Date): end date of the first period
        date_2_start (ee.Date): start date of the second period
        date_2_end (ee.Date): end date of the second period
    
        reduce_method (String): reduction method to reduce probabilities with the following options:
                            - 'mean': takes the mean of the probability ee.Images from start_date to end_date
                            - 'median': takes the median of the probability ee.Images from start_date to end_date
                            Defaults to 'median'
    Returns:
        An ee.Image of probability differences with second_image - first_image, the outputted image has dates set with the second date's range
    """
    if reduce_method == 'mean':
        reducer = ee.Reducer.mean()
    else:
        reducer = ee.Reducer.median()
        
    bandNames = probability_collection.first().bandNames()
    
    first_image = probability_collection.filterDate(date_1_start,date_1_end).reduce(reducer).rename(bandNames)
    second_image = probability_collection.filterDate(date_2_start,date_2_end).reduce(reducer).rename(bandNames)
    output_image = second_image.subtract(first_image)
    output_image = output_image.set('system:time_start',ee.Date(date_2_start))
    output_image = output_image.set('system:time_end',ee.Date(date_2_end))
    return second_image.subtract(first_image)
    
    
def getSeasonalDifference(probability_collection, year, band_names, reduce_method='median', season_list = [['winter',-1,12,1,0,2,'end'],['spring',0,3,1,0,5,'end'],['summer',0,6,1,0,8,'end'],['fall',0,9,1,0,11,'end']], include_difference=True, year_difference=1, image_name='season_probs_{}'):
    """
    Function to convert from daily, monthly, or scene by scene land cover probabilities to seasonal probabilities for year and find the 
        difference in year+1 and year's seasonal probabilities.

    Args:
        probability_collection (ee.ImageCollection): an ImageCollection of images of classified land cover probabilities, each image must have
                                either the start_date and end_date properties defined in order to filter by season definitions
        year (Int): base year to calculate seasonal probabilities
        band_names (List of Strings): image band names to rename after reducing 
        reduce_method (String): reduction method to calculate seasonal probabilities with the following options:
                            - 'mean': takes the mean of the probability ee.Images from start_date to end_date, then takes the ArgMax 
                                    to find the most probable class
                            - 'median': takes the median of the probability ee.Images from start_date to end_date, then takes the ArgMax 
                                    to find the most probable class
                            Defaults to 'median'
        season_list (List): seasons to calculate probabilities for, a 2-dimensional list of size N x 7, where N = number of seasons.
                            The format is [['season_1_name', season_1_start_year_position, season_1_start_month, season_1_start_day, season_1_end_year_position, season_1_end_month, season_1_end_day],
                                            ['season_2_name', season_2_start_year_position, season_2_start_month, season_2_start_day, season_2_end_year_position, season_2_end_month, season_2_end_day]]
                            season_name is used to rename the image for that season
                            season_1_start_position is used to define the year of the start date of the first season, allowing the user to filter to dates including the previous year (for example if December of last year was counted in the current year's winter).
                            season_1_start_month is used to define the month of the start date of the first season
                            season_1_start_day is used to define the day of the start date of the first season, can accept "end" which will calculate the end of the month based on the year (to allow leap year)
                            season_1_end_position is used to define the year of the end date of the first season, allowing the user to filter to dates including the previous year (for example if December of last year was counted in the current year's winter).
                            season_1_end_month is used to define the month of the end date of the first season
                            season_1_end_day is used to define the day of the end date of first the season, can accept "end" which will calculate the end of the month based on the year (to allow leap year)
                            
                            Defaults to:
                                    [['winter',-1,12,1,0,2,'end'],
                                    ['spring',0,3,1,0,5,'end'],
                                    ['summer',0,6,1,0,8,'end'],
                                    ['fall',0,9,1,0,11,'end']]
                            Which translates to 
                                    winter ranging from December of the previous year to the end of February
                                    spring ranging from March to the end of May
                                    summer ranging from June to the end of August
                                    Fall ranging from September to the end of November
    include_difference (Boolean): whether to include the difference from year's seasons to the following year's seasons, defaults to True.
                                    Set to False if you want to only include the current year's season probabilities
    year_difference (Int): if include_difference is True, which year after inputted year to calculate the difference.
                            For example if year_difference=1, then the difference will be calculated as probabilities[year(i+1)]-probabilities[year(i)]
                                        if year_difference=2, then the difference will be calculated as probabilities[year(i+2)]-probabilities[year(i)]
    image_name (String): a name convention for the outputted image, will be formatted with year, defaults to 'season_probs_{year}'
    
    Returns:
        An ee.Image of seasonal probabilities and, if include_difference is True, seasonal differences, with system:index set using the image_name and year
    """
    season_changes = []
    year = int(year)
    for season_definition in season_list:
        season_name = season_definition[0]
        season_name = season_name.lower()
        
        season_start_year_position = season_definition[1]
        season_start_month = season_definition[2]
        season_start_day = season_definition[3]
        season_end_year_position = season_definition[4]
        season_end_month = season_definition[5]
        season_end_day = season_definition[6]
        
        season_start_year_firstYear = year+season_start_year_position
        season_end_year_firstYear = year+season_end_year_position
        
        if include_difference:
            season_start_year_secondYear = year+season_start_year_position+year_difference
            season_end_year_secondYear = year+season_end_year_position+year_difference
        
        if season_start_day == 'end':
            season_firstYear_start_day = calendar.monthrange(season_start_year_firstYear, int(season_start_month))[1]
            if include_difference:
                season_secondYear_start_day = calendar.monthrange(season_end_year_firstYear, int(season_start_month))[1]
        
        else:
            season_firstYear_start_day = season_start_day
            if include_difference:
                season_secondYear_start_day = season_start_day
            
        if season_end_day == 'end':
            season_firstYear_end_day = calendar.monthrange(season_end_year_firstYear, int(season_end_month))[1]
            if include_difference:
                season_secondYear_end_day = calendar.monthrange(season_start_year_secondYear, int(season_end_month))[1]
        
        else:
            season_firstYear_end_day = season_end_day
            if include_difference:
                season_secondYear_end_day = season_end_day
            
        season_firstYear_start = '{}-{}-{}'.format(season_start_year_firstYear, season_start_month, season_firstYear_start_day)
        season_firstYear_end = '{}-{}-{}'.format(season_end_year_firstYear, season_end_month, season_firstYear_end_day)
        
        if include_difference:
            season_secondYear_start = '{}-{}-{}'.format(season_start_year_secondYear, season_start_month, season_secondYear_start_day)
            season_secondYear_end = '{}-{}-{}'.format(season_end_year_secondYear, season_end_month, season_secondYear_end_day)        
        
        if reduce_method=='mean':
            season_image = probability_collection.filterDate(season_firstYear_start,season_firstYear_end).reduce(ee.Reducer.mean()).rename(band_names)
            if include_difference:
                diff_image = getTemporalProbabilityDifference(probability_collection, season_firstYear_start, 
                                                            season_firstYear_end, season_secondYear_start, season_secondYear_end, reduce_method='mean').rename(band_names)
        else:
            season_image = probability_collection.filterDate(season_firstYear_start,season_firstYear_end).reduce(ee.Reducer.median()).rename(band_names)
            if include_difference:
                diff_image = getTemporalProbabilityDifference(probability_collection, season_firstYear_start, 
                                                            season_firstYear_end, season_secondYear_start, season_secondYear_end, reduce_method='median').rename(band_names)
    
        season_image = season_image.set('system:index','{}_start'.format(season_name))
        
        season_changes.append(season_image)
        
        if include_difference:
            diff_image = diff_image.set('system:index','{}_difference'.format(season_name))
            season_changes.append(diff_image)            
        
    season_changes = ee.ImageCollection(season_changes) 
    season_changes = season_changes.toBands()
    season_changes = season_changes.set('system:index',image_name.format(year))
    season_changes = season_changes.set('system:time_start',ee.Date(season_firstYear_start))
    season_changes = season_changes.set('system:time_end',ee.Date(season_firstYear_end))
    return season_changes



def convertDftoFC(feature_collection, property_names):
    """
    Function to convert an ee.FeatureCollection to a pandas.DataFrame

    Args:
        feature_collection (ee.FeatureCollection): an image to sample
        property_names (List): list of feature names to select from feature_collection

    Returns:
        A pandas.DataFrame of feature_collection
    """
    df = pd.DataFrame()
    for property_name in property_names:
        property_values = feature_collection.aggregate_array(property_name).getInfo()
        df[property_name] = property_values
    return df


def convertPointsDfToFc(df,projection=None,lat_name='latitude',lon_name='longitude'):
    """
    Function to convert a DataFrame containing a point locations to an ee.FeatureCollection

    Args:
        df (pandas.DataFrame): an image to sample
        projection (ee.Projection): projection of points
        lat_name (String): column name containing the latitude value
        lon_name (String): column name containing the longitude value

    Returns:
        An ee.FeatureCollection of inputted df
    """
    feature_collection_list = []
    for i,row in df.iterrows():
        geometry = ee.Geometry.Point([row[lon_name],row[lat_name]],projection)
        row_dict = row.to_dict()
        row_feature = ee.Feature(geometry,row_dict)
        feature_collection_list.append(row_feature)
    return ee.FeatureCollection(feature_collection_list)


def convertSeriesToFeature(series,projection='EPSG:4326',lat_name='latitude',lon_name='longitude'):
    """
    Function to convert a Series containing a point location to an ee.Feature

    Args:
        series (pd.Series): an image to sample
        projection (String): projection
        lat_name (String): key containing the latitude value
        lon_name (String): key containing the longitude value

    Returns:
        An ee.Feature containing Series
    """
    geometry = ee.Geometry.Point([series[lon_name],series[lat_name]])#,projection)
    row_dict = series.to_dict()
    row_feature = ee.Feature(geometry,row_dict)
    return row_feature


# #Function to sample image data at point locations (sampleBandPoints) and rename new property to image_name
# def getSampleImageData(image, sampleBandPoints, image_name):
#     sampleImageData = image.reduceRegions(
#         collection=sampleBandPoints,
#         reducer=ee.Reducer.first(),
#         crs=crs,
#         crsTransform=crsTransform
#         )
#     #Rename sampled values from "first" to image_name
#     sampleImageData = sampleImageData.map(lambda x: x.set({image_name:x.get('first')}))
#     return sampleImageData
    

def getStratifiedSampleBandPoints(image, region, bandName, **kwargs):
    """
    Function to perform stratitfied sampling of an image over a given region, using ee.Image.stratifiedSample(image, region, bandName, **kwargs)

    Args:
        image (ee.Image): an image to sample
        region (ee.Geometry): the geometry over which to sample
        bandName (String): the bandName to select for stratification

    Returns:
        An ee.FeatureCollection of sampled points along with coordinates
    """
    dargs = {
        'numPoints': 1000,
        'classBand': bandName,
        'region': region
    }
    dargs.update(kwargs)
    stratified_sample = image.stratifiedSample(**dargs)
    return stratified_sample
    
    
def getSampleBandPoints(image, region, **kwargs):
    """
    Function to perform sampling of an image over a given region, using ee.Image.samp;e(image, region, **kwargs)

    Args:
        image (ee.Image): an image to sample
        region (ee.Geometry): the geometry over which to sample

    Returns:
        An ee.FeatureCollection of sampled points along with coordinates
    """
    dargs = {
        'numPixels': 1000,
        'region': region
    }
    dargs.update(kwargs)
    sample = image.sample(**dargs)
    return sample

def squashScenesToAnnualProbability(probability_collection, years, start_date='{}-01-01', end_date='{}-12-31', method='median',image_name='{}'):
    """
    Function to convert from daily, monthly, or scene by scene land cover classification probabilities images to annual probabilities.

    Args:
        probability_collection (ee.ImageCollection): an Image Collection of classified images where each band represents 
                            the predicted probability of the pixel being in class x. Each image must have the
                            'start_date' and 'end_date' properties set in order to filter by date.
        years (List or numpy.Array): a list or numpy array of years to reduce
        start_date (String): the first day of the year to reduce over, in the format '{}-month-day where {} will be 
                            replaced by the year, defaults to January 1st
        end_date (String): the last day of the year to reduce over, in the format '{}-month-day where {} will be replaced 
                            by the year, defaults to December 31st
        method (String): the method to reduce the Image Collection, with the following options:
                            - 'mean': takes the mean of the probability ee.Images from start_date to end_date, then takes the ArgMax 
                                    to find the most probable class
                            - 'median': takes the median of the probability ee.Images from start_date to end_date, then takes the ArgMax 
                                    to find the most probable class
                            - 'mode': takes ArgMax of every probability ee.Image from start_date to end_date to find the most probable 
                                    class for each ee.Image, then takes the mode. 
                            Defaults to 'median'
        image_name (String): name for outputted image names, images will be named image_name.format(year), defaults to '{}'.format(year)

    Returns:
        An ee.ImageCollection where each image is the annual class probabilities for each year
    """
    predicted_collection = []
    for year in years:
        year = int(year)
        year_probs = probability_collection.filterDate(start_date.format(year),end_date.format(year))
        band_names = year_probs.first().bandNames()
        if method=='mean':
            year_probs = year_probs.reduce(ee.Reducer.mean()).rename(band_names)
        else:
            year_probs = year_probs.reduce(ee.Reducer.median()).rename(band_names)
        year_probs = year_probs.set('system:index',image_name.format(year))        
        year_probs = year_probs.set('system:time_start',ee.Date(start_date.format(year)))
        year_probs = year_probs.set('system:time_end',ee.Date(end_date.format(year)))
        predicted_collection.append(year_probs)
    return ee.ImageCollection(predicted_collection)    
    
    
def squashScenesToAnnualClassification(probability_collection, years, start_date='{}-01-01', end_date='{}-12-31', method='median',image_name='{}'):
    """
    Function to convert from daily, monthly, or scene by scene land cover classification probabilities images to annual classifications.

    Args:
        probability_collection (ee.ImageCollection): an Image Collection of classified images where each band represents 
                            the predicted probability of the pixel being in class x. Each image must have the
                            'start_date' and 'end_date' properties set in order to filter by date.
        years (List or numpy.Array): a list or numpy array of years to reduce
        start_date (String): the first day of the year to reduce over, in the format '{}-month-day where {} will be 
                            replaced by the year, defaults to January 1st
        end_date (String): the last day of the year to reduce over, in the format '{}-month-day where {} will be replaced 
                            by the year, defaults to December 31st
        method (String): the method to reduce the Image Collection, with the following options:
                            - 'mean': takes the mean of the probability ee.Images from start_date to end_date, then takes the ArgMax 
                                    to find the most probable class
                            - 'median': takes the median of the probability ee.Images from start_date to end_date, then takes the ArgMax 
                                    to find the most probable class
                            - 'mode': takes ArgMax of every probability ee.Image from start_date to end_date to find the most probable 
                                    class for each ee.Image, then takes the mode. 
                            Defaults to 'median'
        image_name (String): name for outputted image names, images will be named image_name.format(year), defaults to '{}'.format(year)

    Returns:
        An ee.ImageCollection where each image has a single band "class" with the most likely class for that year
    """
    predicted_collection = []
    for year in years:
        year = int(year)
        year_probs = probability_collection.filterDate(start_date.format(year),end_date.format(year))
        if method=='mean':
            year_probs = year_probs.reduce(ee.Reducer.mean())
            probs_array = year_probs.toArray().toFloat()
            probs_max = probs_array.arrayArgmax().arrayGet(0).add(1)
        elif method=='mode':
            year_maxes = year_probs.map(lambda x: x.toArray().toFloat().arrayArgmax().arrayGet(0).add(1))
            probs_max = year_maxes.reduce(ee.Reducer.mode())
        else:
            year_probs = year_probs.reduce(ee.Reducer.median())
            probs_array = year_probs.toArray().toFloat()
            probs_max = probs_array.arrayArgmax().arrayGet(0).add(1)
        probs_max = probs_max.rename('class')
        probs_max = probs_max.set('system:index',image_name.format(year))
        probs_max = probs_max.set('system:time_start',ee.Date(start_date.format(year)))
        probs_max = probs_max.set('system:time_end',ee.Date(end_date.format(year)))
        
        predicted_collection.append(probs_max)
    return ee.ImageCollection(predicted_collection)
        
        
def probabilityToClassification(image):
    """
    Function to convert an image of land cover classification probabilities to land cover classification

    Args:
        image (ee.Image): an Image where each band represents a land cover classification probability

    Returns:
        An ee.Image with one band representing the land cover classification with the highest probability, each image will have 
                integer values representing the number band that has the highest value. For instance if the third band has the 
                highest probability in pixel[i,j], the outputted value in pixel[i,j] will be 3.
                If there are multiple bands of the same maximum value, returns the first band.
    
    Example: 
        The inputted image has the following bands, Agriculture, Forest, Grassland, Water, Urban, where each  band represents the 
                probability that the pixel is of that classification.
        Say for pixel[i,j] of image k, the bands have the following values:
                Agriculture: 0.1
                Forest: 0.6
                Grassland: 0.2
                Water: 0.5
                Urban: 0.5
        Then the returned value of pixel[i,j] in image k would be 2 as Forest is the second band in the image
        
    """
    #Convert bands to an array
    probs_array = image.toArray().toFloat()
    #Get the argMax to find the band that has the highest probability, add 1 because indices start at 0
    probs_max = probs_array.arrayArgmax().arrayGet(0).add(1)
    return probs_max
    
    
def convertClassificationsToBinaryImages(image, classes_dict):
    """
    Function to convert a single band image of land cover classifications to a multiband image of binary (0,1) bands where each band is the binary

    Args:
        image (ee.Image): an Image with one band that represents the land cover classification
        classes_dict (ee.Dicitonary): a dictionary with keys for land cover class names and values for the land cover class values

    Returns:
        An ee.ImageCollection with each image corresponding to a band from the input image, and multiple bands for binary (0,1) variables representing
                if the pixel was in the band class

    Example:
        image is an ee.Image with bands '2016', '2017', '2018
        classes_dict is an ee.Dictionary({'Agriculture':1, 'Forest':2, 'Grassland':3, 'Water':4, 'Urban':5})
        
        If pixel[i,j] in band '2016' is 2, pixel[i,j] in the image '2016' will have the following band values
            'Agriculture': 0
            'Forest': 1
            'Grassland': 0
            'Water': 0
            'Urban': 0
    """
    def lcYearToBinary(band):
        band_image = image.select([band])
        def lcYearToBinaryNested(key):
            key = ee.String(key)
            class_image = ee.Image.constant(classes_dict.get(key))
            out_image = ee.Image(band_image.eq(class_image))
            return ee.Image(out_image)
        band_collection = ee.ImageCollection(classes_dict.keys().map(lcYearToBinaryNested))
        band_collection = band_collection.toBands().rename(classes_dict.keys())
        band_collection = band_collection.set('system:index',band)
        band_collection = band_collection.set('system:time_start',image.get('system:time_start'))
        band_collection = band_collection.set('system:time_end',image.get('system:time_end'))
        return band_collection

    imageCollection = ee.ImageCollection(image.bandNames().map(lcYearToBinary))
    return imageCollection

