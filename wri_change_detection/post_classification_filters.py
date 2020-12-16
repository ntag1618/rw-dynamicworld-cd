#Import necessary libraries
import os
import ee
import numpy as np
import pandas as pd
import random
from . import preprocessing as npv

"""
This module is designed the implement post-classification filters developed by the MapBiomas (https://mapbiomas.org/) team, including various contributors
to each biome and cross-cutting theme. You can read more of the methodology in the Algorithm Theoretical Basis Document (ATBD) Page (https://mapbiomas.org/en/download-of-atbds)
on their website, including the main ATBD and appendices for each each biome and cross-cutting themes. 

From Section 3.5 of the ATBD, MapBiomas defines post-classification filters,
"[due] to the pixel-based classification method and the long temporal series, a chain of post-classification filters was applied. 
The first post-classification action involves the application of temporal filters. 
Then, a spatial filter was applied followed by a gap fill filter. 
The application of these filters remove classification noise. 
These post-classification procedures were implemented in the Google Earth Engine platform"


MapBiomas provides code for the land cover classification and post-processing in Github, with the home directory here (https://github.com/mapbiomas-brazil/mapbiomas-brazil.github.io)
Some of the code below was taken directly from code written by a Biome Team and converted from JavaScript to Python. In these cases we will link directly
to that page and the lines from which they were taken.
In other cases new code was implemented in order to make the code independent of the inputted land cover classification images.

For each post-classification filter, a description of the filter from the ATBD will be included.

Below is the copy of the licensing for MapBiomas:
The MapBiomas data are public, open and free through license Creative Commons CC-CY-SA and the simple reference of the source observing the following format:
"Project MapBiomas - Collection v5.0 of Brazilian Land Cover & Use Map Series, accessed on 12/14/2020 through the link: https://github.com/mapbiomas-brazil/mapbiomas-brazil.github.io"
"MapBiomas Project - is a multi-institutional initiative to generate annual land cover and use maps using automatic classification processes applied to satellite images. The complete description of the project can be found at http://mapbiomas.org".
Access here the scientific publication: Souza at. al. (2020) - Reconstructing Three Decades of Land Use and Land Cover Changes in Brazilian Biomes with Landsat Archive and Earth Engine - Remote Sensing, Volume 12, Issue 17, 10.3390/rs12172735.
"""

#Section 3.5.3. of the ATBD: Temporal filter
# The temporal filter uses sequential classifications in a three-to-five-years unidirectional moving window to identify temporally non-permitted transitions. Based on generic rules (GR), the temporal filter inspects the central position of three to five consecutive years, and if the extremities of the consecutive years are identical but the centre position is not, then the central pixels are reclassified to match its temporal neighbour class. For the three years based temporal filter, a single central position shall exist, for the four and five years filters, two and there central positions are respectively considered.
# Another generic temporal rule is applied to extremity of consecutive years. In this case, a three consecutive years window is used and if the classifications of the first and last years are different from its neighbours, this values are replaced by the classification of its matching neighbours.
#All code for the Temporal Filters was provided by the Pampa Team (https://github.com/mapbiomas-brazil/pampa) in this file (https://github.com/mapbiomas-brazil/pampa/blob/master/Step006_Filter_03_temporal.js)
#Functions were rewritten in Python and made independent of the land cover classification image.
def mask3(imagem, value, bandNames):
    """
    A helper function to perform a 3 year moving window filter for a single land cover value (such as Forest as 1) for one three year window representing 
    year(i-1), year(i), year(i+1) annual land cover classifications. 
    This function applies on one window, and should only be called using the function applyWindow3years.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The temporal filter inspects the central position of three consecutive years, and if the extremities of the consecutive years are identical 
    but the centre position is not, then the central pixels are reclassified to match its temporal neighbour class.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for a single year
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for 
                                        years: year(i-1), year(i), and year(i+1)

    Returns:
        An ee.Image with one band representing the central year's land cover classification filtered with the 3 year window
    """
    mask = imagem.select(bandNames[0]).eq(value) \
        .bitwiseAnd(imagem.select(bandNames[1]).neq(value)) \
        .bitwiseAnd(imagem.select(bandNames[2]).eq(value))
    change_img = imagem.select(bandNames[1]).mask(mask.eq(1)).where(mask.eq(1), value)
    img_out = imagem.select(bandNames[1]).blend(change_img)
    return img_out
    
    
def mask4(imagem, value, bandNames):
    """
    A helper function to perform a 4 year moving window filter for a single land cover value (such as Forest as 1) for one four year window representing 
    year(i-1), year(i), year(i+1), and year(i+2) annual land cover classifications. 
    This function applies on one window, and should only be called using the function applyWindow4years.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The temporal filter inspects the central position of four consecutive years, and if the extremities of the consecutive years are identical 
    but the centre position is not, then the central pixels are reclassified to match its temporal neighbour class.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for a single year
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for
                                        years: year(i-1), year(i), year(i+1), and year(i+2)

    Returns:
        An ee.Image with one band representing the central year's land cover classification filtered with the 4 year window
    """
    mask = imagem.select(bandNames[0]).eq(value) \
        .bitwiseAnd(imagem.select(bandNames[1]).neq(value)) \
        .bitwiseAnd(imagem.select(bandNames[2]).neq(value)) \
        .bitwiseAnd(imagem.select(bandNames[3]).eq(value)) 
    change_img  = imagem.select(bandNames[1]).mask(mask.eq(1)).where(mask.eq(1), value)
    change_img1 = imagem.select(bandNames[2]).mask(mask.eq(1)).where(mask.eq(1), value) 
    img_out = imagem.select(bandNames[1]).blend(change_img).blend(change_img1)
    return img_out
  

def mask5(imagem, value, bandNames):
    """
    A helper function to perform 5 year moving window filter for a single land cover value (such as Forest as 1) for one five year window representing 
    year(i-1), year(i), year(i+1), year(i+2), and year(i+3) annual land cover classifications. 
    This function applies on one window, and should only be called using the function applyWindow5years.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The temporal filter inspects the central position of five consecutive years, and if the extremities of the consecutive years are identical 
    but the centre position is not, then the central pixels are reclassified to match its temporal neighbour class.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for a single year
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for
                                        years: year(i-1), year(i), year(i+1), year(i+2), and year(i+3)

    Returns:
        An ee.Image with one band representing the central year's land cover classification filtered with the 5 year window
    """
    mask = imagem.select(bandNames[0]).eq(value) \
        .bitwiseAnd(imagem.select(bandNames[1]).neq(value)) \
        .bitwiseAnd(imagem.select(bandNames[2]).neq(value)) \
        .bitwiseAnd(imagem.select(bandNames[3]).neq(value)) \
        .bitwiseAnd(imagem.select(bandNames[4]).eq(value))
    change_img  = imagem.select(bandNames[1]).mask(mask.eq(1)).where(mask.eq(1), value)  
    change_img1 = imagem.select(bandNames[2]).mask(mask.eq(1)).where(mask.eq(1), value)  
    change_img2 = imagem.select(bandNames[3]).mask(mask.eq(1)).where(mask.eq(1), value)  
    img_out = imagem.select(bandNames[1]).blend(change_img).blend(change_img1).blend(change_img2)
    return img_out


def applyWindow5years(imagem, value, bandNames):
    """
    Function to perform a 5 year moving window filter for a single land cover value (such as Forest as 1) for all years in an image.
    Calls the function mask5.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The temporal filter inspects the central position of five consecutive years, and if the extremities of the consecutive years are identical 
    but the centre position is not, then the central pixels are reclassified to match its temporal neighbour class.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for all years
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image with year(0) through year(-3) filtered using the 5 year moving window, and the remaining three years unchanged
    """
    img_out = imagem.select(bandNames[0])
    for i in np.arange(1, len(bandNames)-3):
        img_out = img_out.addBands(mask5(imagem, value,bandNames[(i-1):(i+4)]))
    img_out = img_out.addBands(imagem.select(bandNames[-3]))
    img_out = img_out.addBands(imagem.select(bandNames[-2]))
    img_out = img_out.addBands(imagem.select(bandNames[-1]))
    return img_out  
    
    
def applyWindow4years(imagem, value, bandNames):
    """
    Function to perform a 4 year moving window filter for a single land cover value (such as Forest as 1) for all years in an image.
    Calls the function mask4.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The temporal filter inspects the central position of four consecutive years, and if the extremities of the consecutive years are identical 
    but the centre position is not, then the central pixels are reclassified to match its temporal neighbour class.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for all years
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image with year(0) through year(-2) filtered using the 4 year moving window, and the remaining two years unchanged
    """
    img_out = imagem.select(bandNames[0])
    for i in np.arange(1, len(bandNames)-2):
        img_out = img_out.addBands(mask4(imagem, value,bandNames[(i-1):(i+3)]))
    img_out = img_out.addBands(imagem.select(bandNames[-2]))
    img_out = img_out.addBands(imagem.select(bandNames[-1]))
    return img_out        


def applyWindow3years(imagem, value, bandNames):
    """
    Function to perform a 3 year moving window filter for a single land cover value (such as Forest as 1) for all years in an image.
    Calls the function mask3.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The temporal filter inspects the central position of three consecutive years, and if the extremities of the consecutive years are identical 
    but the centre position is not, then the central pixels are reclassified to match its temporal neighbour class.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for all years
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image with year(0) through year(-1) filtered using the 3 year moving window, and the remaining final year unchanged
    """
    img_out = imagem.select(bandNames[0])
    for i in np.arange(1, len(bandNames)-1):
        img_out = img_out.addBands(mask3(imagem, value,bandNames[(i-1):(i+2)]))
    img_out = img_out.addBands(imagem.select(bandNames[-1]))
    return img_out        


def applyMask3first(imagem, value, bandNames):
    """
    Function to perform a 3 year window filter for a single land cover value (such as Forest as 1) for the first year in an image.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    For the first year of land cover classifications, a three consecutive years window is used and if the classifications of the 
    first and last years are different from its neighbours, this values are replaced by the classification of its matching neighbours.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for all years
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image with year(0) filtered using the 3 year window, and the remaining years unchanged
    """
    mask = imagem.select(bandNames[0]).neq(value) \
        .bitwiseAnd(imagem.select(bandNames[1]).eq(value)) \
        .bitwiseAnd(imagem.select(bandNames[2]).eq(value))
    change_img = imagem.select(bandNames[0]).mask(mask.eq(1)).where(mask.eq(1), value)
    img_out = imagem.select(bandNames[0]).blend(change_img)
    img_out = img_out.addBands(imagem.select(bandNames[1:]))
    return img_out


def applyMask3last(imagem, value, bandNames):
    """
    Function to perform a 3 year window filter for a single land cover value (such as Forest as 1) for the final year in an image.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    For the first year of land cover classifications, a three consecutive years window is used and if the classifications of the 
    first and last years are different from its neighbours, this values are replaced by the classification of its matching neighbours.
    This function can be applied to whichever land cover values the user decides, such as all of the land cover values or a select few.

    Args:
        imagem (ee.Image): an Image where each band represents a land cover classification for all years
        value (Int): an Integer representing the land cover class value to filter
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image with year(-1) filtered using the 3 year window, and the remaining years unchanged
    """
    mask = imagem.select(bandNames[-3]).eq(value) \
        .bitwiseAnd(imagem.select(bandNames[-2]).eq(value)) \
        .bitwiseAnd(imagem.select(bandNames[-1]).neq(value))
    change_img = imagem.select(bandNames[-1]).mask(mask.eq(1)).where(mask.eq(1), value)
    img_out = imagem.select(bandNames[0:-1])
    img_out = img_out.addBands(imagem.select(bandNames[-1]).blend(change_img))
    return img_out


#Section 3.5.2. of the ATBD: Spatial filter:
#Spatial filter was applied to avoid unwanted modifications to the edges of the pixel groups (blobs), a spatial filter was built based on the “connectedPixelCount” function. Native to the GEE platform, this function locates connected components (neighbours) that share the same pixel value. Thus, only pixels that do not share connections to a predefined number of identical neighbours are considered isolated. In this filter, at least five connected pixels are needed to reach the minimum connection value. Consequently, the minimum mapping unit is directly affected by the spatial filter applied, and it was defined as 5 pixels (~0.5 ha).
#All code for the spatial filter was provided within the intregration-toolkit (https://github.com/mapbiomas-brazil/integration-toolkit) that is used to combine land cover classifications from each biome and cross-cutting theme team. The direct code was provided in this file (https://github.com/mapbiomas-brazil/integration-toolkit/blob/master/mapbiomas-integration-toolkit.js)
#Functions were rewritten in Python and made independent of the land cover classification image.

def majorityFilter(image, params):
    """
    A helper function to perform a spatial filter based on connectedPixelCount for one land cover class value. 
    Spatial filter was applied to avoid unwanted modifications to the edges of the pixel groups (blobs), a spatial filter was built 
    based on the “connectedPixelCount” function. Native to the GEE platform, this function locates connected components (neighbours) 
    that share the same pixel value. Thus, only pixels that do not share connections to a predefined number of identical 
    neighbours are considered isolated. In this filter, at least some number of connected pixels are needed to reach the minimum connection value, 
    defined in params.
    
    Args:
        image (ee.Image): an Image where each band represents a land cover classification for a each year
        params (Dictionary): a dictionary of the form {'classValue': [Int], 'maxSize': [Int]}, where classValue is the value of the land cover
                                class to filter, and the maxSize is the minimum number of connected pixels of the same land cover class
                                to not be removed.
                                If the number of connectedPixels of that land cover class for the central pixels is less than maxSize,
                                the central pixel is replaced by the mode of the surrounding pixels, defined by a square kernel with size shift 1.
                                # Squared kernel with size shift 1:
                                # [[p(x-1,y+1), p(x,y+1), p(x+1,y+1)]
                                # [ p(x-1,  y), p( x,y ), p(x+1,  y)]
                                # [ p(x-1,y-1), p(x,y-1), p(x+1,y-1)]
    Returns:
        A multiband ee.Image with bands representing years of land cover classifications, filtered with the spatial filter to remove clusters less than
            the inputted size in 'maxSize' of the params dictionary for the class 'classValue' defined in the params dictionary
    
    Example: 
        The inputted image has the following bands, classification_1985, classification_1986, classification_1987 representing the classifcations
                    for 1985, 1986, 1987.
        The params dictionary is {'classValue': 1, 'maxSize': 5}
    
        Say for pixel[i,j] of image k, pixel[i,j] has the value 1, and connectedPixelCount finds that it only has 4 connected neighbors of the same class.
        Then the returned value of pixel[i,j] in image k would be the mode of the surrounding pixels, defined by a square kernel with size shift 1.
    
    """
    params = ee.Dictionary(params)
    maxSize = ee.Number(params.get('maxSize'))
    classValue = ee.Number(params.get('classValue'))
    
    #Generate a mask from the class value
    classMask = image.eq(classValue)
    
    #Labeling the group of pixels until 100 pixels connected
    labeled = classMask.mask(classMask).connectedPixelCount(maxSize, True)
    
    #Select some groups of connected pixels
    region = labeled.lt(maxSize)
    
    # Squared kernel with size shift 1
    # [[p(x-1,y+1), p(x,y+1), p(x+1,y+1)]
    # [ p(x-1,  y), p( x,y ), p(x+1,  y)]
    # [ p(x-1,y-1), p(x,y-1), p(x+1,y-1)]
    kernel = ee.Kernel.square(1)
    
    #Find neighborhood
    neighs = image.neighborhoodToBands(kernel).mask(region)

    #Reduce to majority pixel in neighborhood
    majority = neighs.reduce(ee.Reducer.mode())
    
    #Replace original values for new values
    filtered = image.where(region, majority)
    
    return ee.Image(filtered)
    
def applySpatialFilter(image,filterParams):
    """
    Function to perform a spatial filter based on connectedPixelCount for land cover class values defined in filterParams. 
    Calls the function majorityFilter.
    Spatial filter was applied to avoid unwanted modifications to the edges of the pixel groups (blobs), a spatial filter was built 
    based on the “connectedPixelCount” function. Native to the GEE platform, this function locates connected components (neighbours) 
    that share the same pixel value. Thus, only pixels that do not share connections to a predefined number of identical 
    neighbours are considered isolated. In this filter, at least some number of connected pixels are needed to reach the minimum connection value, 
    defined in params.

    Args:
        image (ee.Image): an Image where each band represents a land cover classification for a each year
        params (List of Dictionaries): a List of Dictionaries of the form 
                                        [{'classValue': [Int], 'maxSize': [Int]},
                                        {'classValue': [Int], 'maxSize': [Int]},
                                        {'classValue': [Int], 'maxSize': [Int]}]
                                        where classValue is the value of the land cover to filter and maxSize is the minimum number of connected pixels 
                                        of the same land cover class to not be removed.
                                If the number of connectedPixels of that land cover class for the central pixels is less than maxSize,
                                the central pixel is replaced by the mode of the surrounding pixels, defined by a square kernel with size shift 1.
                                # Squared kernel with size shift 1:
                                # [[p(x-1,y+1), p(x,y+1), p(x+1,y+1)]
                                # [ p(x-1,  y), p( x,y ), p(x+1,  y)]
                                # [ p(x-1,y-1), p(x,y-1), p(x+1,y-1)]
    Returns:
        A multiband ee.Image with bands representing years of land cover classifications, filtered with the spatial filter to remove clusters less than
            the inputted size in 'maxSize' of the params dictionary for the class 'classValue' defined in the params dictionary for each params dictionary
            in the list of filterParams.
    
    Example: 
        The inputted image has the following bands, classification_1985, classification_1986, classification_1987 representing the classifcations
                    for 1985, 1986, 1987.
        The filterParams list of dictionaries is:
                filterParams = [
                    {'classValue': 1, 'maxSize': 3},
                    {'classValue': 2, 'maxSize': 5},
                    {'classValue': 3, 'maxSize': 5},
                    {'classValue': 4, 'maxSize': 3},
                    {'classValue': 5, 'maxSize': 3},
                    {'classValue': 6, 'maxSize': 3},
                    {'classValue': 7, 'maxSize': 3},
                    {'classValue': 8, 'maxSize': 3},
                    {'classValue': 9, 'maxSize': 3},
                ]
    """
    #Loop through list of parameters and apply spatial filter using majorityFilter
    for params in filterParams:
        image = majorityFilter(ee.Image(image),params)
    return image


#Section 3.5.1. of the ATBD: Gap fill:
#The Gap fill filter was used to fill possible no-data values. In a long time series of severely cloud-affected regions, it is expected that no-data values may populate some of the resultant median composite pixels. In this filter, no-data values (“gaps”) are theoretically not allowed and are replaced by the temporally nearest valid classification. In this procedure, if no “future” valid position is available, then the no-data value is replaced by its previous valid class. Up to three prior years can be used to fill in persistent no-data positions. Therefore, gaps should only exist if a given pixel has been permanently classified as no-data throughout the entire temporal domain.
#All code for the Gap Filters was provided by the Pampa Team (https://github.com/mapbiomas-brazil/pampa) in this file (https://github.com/mapbiomas-brazil/pampa/blob/master/Step006_Filter_01_gagfill.js), although the same gap fill is applied to all cross-cutting themes
#and biome groups.
#Functions were rewritten in Python and made independent of the land cover classification image. The implementation of the gap fill in the MapBiomas
#code actually applies both a forward no-data filter and a backwards no-data filter. 

def applyForwardNoDataFilter(image, bandNames):
    """
    Function to perform a forward moving gap fill for all years in an image.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The forward gap fill is applied iteratively from the first year of bandNames through the final year, where if the current image has
    missing data, it is filled with the following year's values.
    Args:
        image (ee.Image): an Image where each band represents a land cover classification for all years
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image, one band for each year, with the forward gap fill applied to all years, and bands returned in chronological order.
    """
    #Get a list of band names from year(1) through the last year
    bandNamesEE = ee.List(bandNames[1:])
    
    #Define forwards filter
    #In first iteration, bandName=bandNames[1] and previousImage is image.select(bandNames[0]), or the classifications for the first year
    #currentImage = image.select(bandNames[1]), the image for the second year
    #previousImage = image.select(bandNames[0]), the first year
    #Find where the second year has missing data, replace those values with the values of the first year
    #Append previousImage to currentImage, so now currentImage is a two band image, with the first band being the second year with the gap fill
    #and the second band is the first years classification
    #The iteration continues, now with followingImage.select[0] being the second year with the gap fill applied, and bandName is the third year
    def forwardNoDataFilter(bandName, previousImage):
        currentImage = image.select(ee.String(bandName))
        previousImage = ee.Image(previousImage)
        currentImage = currentImage.unmask(previousImage.select([0]))
        return currentImage.addBands(previousImage)
    
    #Iterate through all the years, starting with the first year's classification
    filtered = bandNamesEE.iterate(forwardNoDataFilter,ee.Image(image.select(bandNames[0])))
    filtered = ee.Image(filtered)
    return filtered.select(bandNames)


def applyBackwardNoDataFilter(image, bandNames):
    """
    Function to perform a backward moving gap fill for all years in an image.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    The backward gap fill is applied iteratively from the last year of bandNames through the first year, where if the current image has
    missing data, it is filled with the previous year's values.
    Args:
        image (ee.Image): an Image where each band represents a land cover classification for all years
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image, one band for each year, with the forward gap fill applied to all years, and bands returned in chronological order.
    """
    #Get a list of band names to iterate over, from year(-2) through year(0)
    bandNamesEE = ee.List(bandNames[:-1]).reverse()
    
    #Define backwards filter
    #In first iteration, bandName=bandNames[-2] and followingImage is image.select(bandNames[-1]), or the classifications for the final year
    #currentImage = image.select(bandNames[-2]), the second to last year
    #followingImage = image.select(bandNames[-1]), the final year
    #Find where the second to last year has missing data, replace those values with the values of the following year
    #Append followingImage to currentImage, so now currentImage is a two band image, with the first band being the second to last year with the gap fill
    #and the second band is the final years classification
    #The iteration continues, now with followingImage.select[0] being the second to last year with the gap fill applied, and bandName is the third to last year
    def backwardNoDataFilter(bandName, followingImage):
        currentImage = image.select(ee.String(bandName))
        followingImage = ee.Image(followingImage)
        currentImage = currentImage.unmask(followingImage.select([0]))
        return currentImage.addBands(followingImage)
        
    #Apply backwards filter, starting with the final year and iterating through to year(0) 
    filtered = bandNamesEE.iterate(backwardNoDataFilter,ee.Image(image.select(bandNames[-1])))
    #Re-order bands to be in chronological order
    filtered = ee.Image(filtered)
    return filtered.select(bandNames)
    
def applyGapFilter(image, bandNames):
    """
    Function to apply forward gap filling and backward gap filling to an image.
    The image bands do not need to be in order, but the bandNames argument must be in chronological order.
    This funciton calls applyForwardNoDataFilter then applyBackwardNoDataFilter
    Args:
        image (ee.Image): an Image where each band represents a land cover classification for all years
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A multiband ee.Image, one band for each year, with the forward gap fill applied to all years, then the backward gap fill applied to all years,
            and the bands returned in chronological order.
    """
    filtered = applyForwardNoDataFilter(image, bandNames)
    filtered = applyBackwardNoDataFilter(image, bandNames)
    return filtered

    

#Section 3.5.5. of the ATBD: Incident Filter
# An incident filter were applied to remove pixels that changed too many times in the 34 years of time spam. All pixels that changed more than eight times and is connected to less than 6 pixels was replaced by the MODE value of that given pixel position in the stack of years. This avoids changes in the border of the classes and helps to stabilize originally noise pixel trajectories. Each biome and cross-cutting themes may have constituted customized applications of incident filters, see more details in its respective appendices.
#This was not clearly implemented in the MapBiomas code, so this filter was coded by Kristine Lister.

def calculateNumberOfChanges(image, bandNames):
    """
    Function to calculate the total number of times a pixel changed classes across the time series
    Args:
        image (ee.Image): an Image where each band represents a land cover classification for all years
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year

    Returns:
        A single band ee.Image, representing the number of times a pixel changed classes across the time series
    """
    #Get a collection of images where each image has 2 bands: classifications for year(i) and classifications for year(i+1)
    lc_one_change_col = npv.getYearStackIC(image,bandNames, band_indices=[0,1])
    #Get a collection of images where each image represents whether there was change from year(i) to year(i+1) and convert to an image
    lc_one_change_col = lc_one_change_col.map(npv.LC_OneChange)
    lc_one_change_image = lc_one_change_col.toBands()
    #Calculate the number of changes by applying the sum reducer
    lc_sum_changes = lc_one_change_image.reduce(ee.Reducer.sum().unweighted())
    return lc_sum_changes
    
def applyIncidenceFilter(image, bandNames, classDictionary, numChangesCutoff = 8, connectedPixelCutoff=6):
    """
    Function to apply an incidence filter. The incidence filter finds all pixels that changed more than numChangesCutoff times and is connected 
    to less than connectedPixelCutoff pixels, then replaces those pixels with the MODE value of that given pixel position in the stack of years.
    Args:
        image (ee.Image): an Image where each band represents a land cover classification for all years
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year
        classDictionary (ee.Dictionary): an ee.Dictionary where the keys are the land cover classification names and the values are the
                                            land cover classification values.
                                            ee.Dictionary({
                                                        'class_1': int_1, 
                                                        'class_2': int_2, 
                                                        'class_3': int_3, 
                                                        'class_4': int_4, 
                                                        'class_5': int_5})
        numChangesCutoff (Int): the maximum number of changes allowed across the time series. 
        connectedPixelCutoff (Int): the minimum number of connected pixels 
        If the number of times a pixel changes across the time series is GREATER THAN numChangesCutoff AND the number of connectedPixels of the same
                land cover type is LESS THAN connectedPixelCutoff, then the pixel for that year is replaced with the mode of all the years.
        The connectedPixelCutoff is applied to each year, therefore if the number of changes is greater than numChangesCutoff, but the number of 
        connectedPixels is greater than connectedPixelCutoff in year(i), then the pixel in year(i) is not changed. However if the the number of 
        connectedPixels is less than connectedPixelCutoff in year(i+1), then the pixel in year(i+1) is replaced with the mode of the time series in that pixel.
    Returns:
        A multiband ee.Image, one band for each year, with the incidence filter applied to all years.
    """
    #Calculate the number of times a pixel changes throughout the time series and determine if it is over the numChangesCutoff
    num_changes = calculateNumberOfChanges(image, bandNames)
    too_many_changes = num_changes.gt(numChangesCutoff)
    
    #Get binary images of the land cover classifications for the current year
    binary_class_images = npv.convertClassificationsToBinaryImages(image, classDictionary)
    
    #Calculate the number of connected pixels for each land cover class and year, reduce to a single band image representing the number
    #of connected pixels of the same land cover class as the central pixel, and determine if it is over the connectedPixelCutoff
    connected_pixel_count = ee.ImageCollection(binary_class_images.map(lambda x: x.mask(x).connectedPixelCount(100,False).reduce(ee.Reducer.sum()).lt(connectedPixelCutoff)))
    
    #Get a bitwiseAnd determination if the number of connected pixels <= connectedPixelCutoff and the number of changes > numChangesCutoff 
    incidence_filter = ee.ImageCollection(connected_pixel_count.map(lambda x: x.bitwiseAnd(too_many_changes))).toBands().rename(bandNames)
    
    #Get an image that represents the mode of the land cover classes in each pixel
    mode_image = image.reduce(ee.Reducer.mode())
    
    #Replace pixels of image where incidence_filter is True with mode_image
    incidence_filtered = image.where(incidence_filter, mode_image)
    
    return incidence_filtered


#Section 3.5.4. of the ATBD: Frequency Filter  
# This filter takes into consideration the occurrence frequency throughout the entire time series. Thus, all class occurrence with less than given percentage of temporal persistence (eg. 3 years or fewer out of 33) are filtered out. This mechanism contributes to reducing the temporal oscillation associated to a given class, decreasing the number of false positives and preserving consolidated trajectories. Each biome and cross-cutting themes may have constituted customized applications of frequency filters, see more details in their respective appendices.

def applyFrequencyFilter(image, yearBandNames, classDictionary, filterParams):
    """
    Function to apply an frequency filter. This filter takes into consideration the occurrence frequency throughout the entire time series. 
    Thus, all class occurrence with less than given percentage of temporal persistence (eg. 3 years or fewer out of 33) are replaced with the 
    MODE value of that given pixel position in the stack of years.
    
    Args:
        image (ee.Image): an Image where each band represents a land cover classification for all years
        bandNames (List of Strings): a list of band names representing the annual classiciations in chronological order for each year
        classDictionary (ee.Dictionary): an ee.Dictionary where the keys are the land cover classification names and the values are the
                                            land cover classification values.
                                            ee.Dictionary({
                                                        'class_1': int_1, 
                                                        'class_2': int_2, 
                                                        'class_3': int_3, 
                                                        'class_4': int_4, 
                                                        'class_5': int_5})
        filterParams (ee.Dictionary): an ee.Dictionary where the keys are the land cover classification names and the values are the
                                            minimum number of occurances needed in the time series in order to not be filtered out
                                            ee.Dictionary({
                                                        'class_1': int_1, 
                                                        'class_2': int_2, 
                                                        'class_3': int_3, 
                                                        'class_4': int_4, 
                                                        'class_5': int_5})
    
        If the number of occurences of class_1 is LESS THAN int_1 in pixel[i,j], then all occurences of class_1 in pixel[i,j] through the time series
        is replaced by the mode of the land cover classes pixel[i,j] across the time series.
    Returns:
        A multiband ee.Image, one band for each year, with the frequency filter applied to all years.
    """    
    #Grab land cover classes as a list of strings
    lc_classes = classDictionary.keys().getInfo()
    
    #Get binary images of the land cover classifications for the current year
    binary_class_images = npv.convertClassificationsToBinaryImages(image, classDictionary)
    
    #Get the frequency of each class through the years by reducing the image collection to an image using the sum reducer
    class_frequency = binary_class_images.reduce(ee.Reducer.sum().unweighted()).rename(lc_classes)
    
    #Get an image that represents the mode of the land cover classes in each pixel
    mode_image = image.reduce(ee.Reducer.mode())
    
    #Define an image to add bands with frequency filter applied
    out_img = ee.Image()
    
    #Loop through years
    for yearBand in yearBandNames:
        #Select the target year from the image
        yearImage = image.select(yearBand)
        
        #Loop through land cover classes in filterParams
        for lc_class in lc_classes:
            #Get the minimum occurance allowed in that land cover class
            min_occurance = filterParams.get(lc_class)
            
            #Find if the land cover class had less than the number of min_occurances in each pixel
            change_class = class_frequency.select(lc_class).lt(ee.Image.constant(min_occurance))
        
            #If change_class==1, then replace that pixel with the mode of all the years in that pixel
            #This filter is only applied to pixels of this land cover class
            #First mask yearImage to pixels of this land cover class, then get the union of pixels where change_class==1,
            #if both conditions are true, then the pixel is replaced with the mode
            yearImage = yearImage.where(yearImage.eq(ee.Number(classDictionary.get(lc_class))).bitwiseAnd(change_class),mode_image)
        #Rename yearImage to bandName
        yearImage = yearImage.rename(yearBand)
        #Append to output image
        out_img = out_img.addBands(yearImage)
    
    return out_img
    
    
def applyProbabilityCutoffs(imageCollection, params):
    """
    Function to apply a probability filter to land cover probabilities in each image of imageCollection. 
    The user defines which classes will be filtered and how to filter them in the params list.
    The params list is a list of dictionaries, one for each class the user wants to filter.
    The dictionaries in the params list is of the form {'class_name': String, 'class_value': Int, 'filter': String, 'threshold', Float}
    
    If the filter is 'gt' or 'gte' (representing 'greater than' or 'greater than or equal to'), and if the pixel class probability greater than or greater 
        than or equal to the threshold, then final classification is replaced by the value of that class.
    If the filter is 'lt' or 'lte' (representing 'less than' or 'less than or equal to'), and if the pixel class probability less than or less than or
        equal to the threshold, and the pixel is in that class then final classification, then the final classification is replaced by
        the majority class of the neighborhood, where the neighborhood is a square kernel of size 1.
    
    Args:
        imageCollection (ee.ImageCollection): an ee.imageCollection where each image is the land cover classification probabilities (one band for each class)
        params (List of Dictionaries): a List of Dictionaries of the form 
                                        [{'class_name': String, 'class_value': Int, 'filter': String, 'threshold', Float},
                                          {'class_name': String, 'class_value': Int, 'filter': String, 'threshold', Float}]
                                        where:
                                            'class_name' is the name of the land cover class, 
                                            'class_value' is the value of the land cover class
                                            'filter' is "gt", "gte", "lte", or "lt", representing greater than, greater than or equal to, less than or equal to,
                                                        or less than.
                                                        If filter is not in ['gt','gte','lte','lt'] then 'lt' is applied by default.
                                            'threshold' is the threshold to determine whether to filter the pixel
    Say for pixel[i,j] of image k, the bands have the following values:
            Agriculture: 0.1
            Forest: 0.6
            Grassland: 0.2
            Water: 0.5
            Urban: 0.5
    
    Returns:
        An ee.ImageCollection, with each image having one band representing the land cover classification after the filters have been applied.
    """      
        
    #Define function to map across imageCollection
    def probabilityFilter(image):
        
        #Get the classifications from the class with the highest probability
        classifications = npv.probabilityToClassification(image)
        
        #Loop through parameters
        for param in params:
            #Load parameter values
            class_name = param.get('class_name')
            class_value = param.get('class_value')
            filter_name = param.get('filter')
            threshold = param.get('threshold')
            
            if filter_name=='gt':
                #Find where the class_name is greater than threshold
                prob_mask = image.select(class_name).gt(ee.Image.constant(threshold))
                #Replace those pixels with the class value
                classifications = classifications.where(prob_mask,class_value)
            
            elif filter_name=='gte':
                #Find where the class_name is greater than or equal to threshold
                prob_mask = image.select(class_name).gte(ee.Image.constant(threshold))
                #Replace those pixels with the class value
                classifications = classifications.where(prob_mask,class_value)
                
            elif filter_name == 'lte':
                #Find where the class_name is less than or equal to threshold
                prob_mask = image.select(class_name).lte(ee.Image.constant(threshold))
                #Find where classifications are equal to class value
                class_mask = classifications.eq(class_value)
                #We only want to replace pixels where the class probability<=threshold AND classification==class_value
                reclass_mask = prob_mask.bitwiseAnd(class_mask)
        
                #Define square kernel of surrounding pixels
                kernel = ee.Kernel.square(1)
                #Convert to a multiband image, one band for each neighbor
                neighs = classifications.neighborhoodToBands(kernel)
                #Reduce to find the majority class in neighborhood
                majority = neighs.reduce(ee.Reducer.mode())
        
                #Replace pixels where the class probability<=threshold AND classification==class_value with the neighborhood majority class
                classifications = classifications.where(reclass_mask,majority)
        
            else:
                #Find where the class_name is less than or equal to threshold
                prob_mask = image.select(class_name).lt(ee.Image.constant(threshold))
                #Find where classifications are equal to class value
                class_mask = classifications.eq(class_value)
                #We only want to replace pixels where the class probability<=threshold AND classification==class_value
                reclass_mask = prob_mask.bitwiseAnd(class_mask)
        
                #Define square kernel of surrounding pixels
                kernel = ee.Kernel.square(1)
                #Convert to a multiband image, one band for each neighbor
                neighs = classifications.neighborhoodToBands(kernel)
                #Reduce to find the majority class in neighborhood
                majority = neighs.reduce(ee.Reducer.mode())
        
                #Replace pixels where the class probability<=threshold AND classification==class_value with the neighborhood majority class
                classifications = classifications.where(reclass_mask,majority)
        
        return ee.Image(classifications)
    return ee.ImageCollection(imageCollection.map(probabilityFilter))
        
    
    
    
    
    
    
