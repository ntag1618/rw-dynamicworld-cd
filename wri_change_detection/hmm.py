
import ee
import requests
import numpy as np
import cmocean
from rasterio.plot import show_hist, show
import pandas as pd
from sklearn import metrics
import matplotlib.colors
import matplotlib.pyplot as plt


DW_CLASS_LIST = ['water','trees','grass','flooded_vegetation','crops','scrub','built_area','bare_ground','snow_and_ice']
DW_CLASS_COLORS = ["419BDF", "397D49", "88B053", "7A87C6", "E49635", "DFC35A", "C4281B", "A59B8F", "B39FE1"]

change_detection_palette = ['#ffffff', # no_data=0
                              '#419bdf', # water=1
                              '#397d49', # trees=2
                              '#88b053', # grass=3
                              '#7a87c6', # flooded_vegetation=4
                              '#e49535', # crops=5
                              '#dfc25a', # scrub_shrub=6
                              '#c4291b', # builtup=7
                              '#a59b8f', # bare_ground=8
                              '#a8ebff', # snow_ice=9
                              '#616161', # clouds=10
]
statesViz = {'min': 0, 'max': 10, 'palette': change_detection_palette}

boundary_scale = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.6, 9.5, 10.5] #bin edges
dw_class_cmap=matplotlib.colors.ListedColormap(change_detection_palette)
dw_norm=matplotlib.colors.BoundaryNorm(boundary_scale, len(change_detection_palette))
labels_dict = {
    0:'no_data',
    1:'water',
    2:'trees',
    3:'grass',
    4:'flooded_veg',
    5:'crops',
    6:'scrub',
    7:'builtup',
    8:'bare_ground',
    9:'snow_ice',
    10:'clouds'
}
classes_dict = {
    0:'water',
    1:'trees',
    2:'grass',
    3:'flooded_vegetation',
    4:'crops',
    5:'scrub',
    6:'built_area',
    7:'bare_ground',
    8:'snow_and_ice'
}

def retrieve_filtered_img(collection, year, target_img, reducer=ee.Reducer.mean(),scale=10):
    """
    Function to filter, clip, and reduce a collection

    Args:
        collection (ee.imagecollection.ImageCollection): image to download
        year (int or str): Year to pass to an ee filterDate function
        target_img (ee.image.Image): Clip the collection geometry to this image geometry
        reducer (ee.Reducer): reducer method to perform on the collection
        scale (int): scale in meters

    Returns:
        out_image (ee.image.Image): date filtered, geometry clipped image
    """
    out_image = collection.filterDate(
        f'{year}-01-01', f'{year}-12-31').reduce(
        reducer).clipToBoundsAndScale(
        geometry=target_img.geometry(),scale=scale).toFloat()
    return out_image

def stream_file_to_disc(url, filename):
    """
    Function to instantiate a Requests session to facilitate a download from url

    Args:
        url (string): url to initiate download
        filename (string): file path and file name to write download stream to

    Returns:
        Writes download to the specified location
    """
    with requests.Session() as session:
        get = session.get(url, stream=True)
        if get.status_code == 200:
            #print('downloading')
            with open(filename, 'wb') as f:
                for chunk in get.iter_content(chunk_size=1024):
                    f.write(chunk)
        else:
            print(get.status_code)

def write_block(img, dst_filename, temp_dir):
    """
    Function to download specified ee image

    Args:
        img (ee.image.Image): image to download
        dst_filename (String): file name to write to
        temp_dir (tempfile.TemporaryDirectory): will be used as os path prefix to `dst_filename` 
            arg to construct path

    Returns:
        Constructs download url for specified img, downloads geotiff to destination
    """
    url = ee.data.makeThumbUrl(ee.data.getThumbId({'image':img, 'format':'geotiff'}))
    filepath = f'{temp_dir}/{dst_filename}'
    stream_file_to_disc(url, filepath)



def calc_annual_change(model, years):
    """
    Function to calculate annual total class change between years

    Args:
        model (dict): year ints as keys and rasterio dataset pointers as values
        years (list): years to loop through for calculating annual change

    Returns:
        Prints annual fraction of sample pixels changing classes 
    """
    for i in years[:-1]:
        y0 = model[f'{i}'].read().argmax(axis=0).ravel()
        y1 = model[f'{i+1}'].read().argmax(axis=0).ravel()
        accuracy_score = metrics.accuracy_score(y0,y1)
        yr_change = 1-accuracy_score
        print(f'{i}-{i+1}: {yr_change:.2f}')
        
def show_year_diffs_and_classes(dwm, hmm, years, cmap='gray'):
    """
    Function to plot change layer between predicted labels for each year

    Args:
        dwm (dict): Dynamic World Model outputs, with year ints as keys and rasterio dataset pointers as values
        hmm (dict): Hidden Markov Model outputs, with year ints as keys and rasterio dataset pointers as values
        years (list): years to loop through for calculating annual change
        cmap (cmap): colormap

    Returns:
        Plots change layer between predicted class labels for each year
    """
    fig, axs = plt.subplots(len(years)-1,6,figsize=(6*3,(len(years)-1)*3))
    for i,x in enumerate(years[:-1]):
        show(dwm[f'{x}'].read().argmax(axis=0)+1,ax=axs[i,0],
             cmap=dw_class_cmap,norm=dw_norm,title=f'dwm {x}')
        show(np.equal(dwm[f'{x}'].read().argmax(axis=0),dwm[f'{x+1}'].read().argmax(axis=0)),
             ax=axs[i,1], cmap=cmap, title=f'dwm {x}-{x+1}')
        show(dwm[f'{x+1}'].read().argmax(axis=0)+1,ax=axs[i,2],
             cmap=dw_class_cmap,norm=dw_norm,title=f'dwm {x+1}')
        show(hmm[f'{x}'].read().argmax(axis=0)+1,ax=axs[i,3],
             cmap=dw_class_cmap,norm=dw_norm,title=f'hmm {x}')
        show(np.equal(hmm[f'{x}'].read().argmax(axis=0),hmm[f'{x+1}'].read().argmax(axis=0)), 
             ax=axs[i,4], cmap=cmap,title=f'hmm {x}-{x+1}')
        show(hmm[f'{x+1}'].read().argmax(axis=0)+1,ax=axs[i,5],
             cmap=dw_class_cmap,norm=dw_norm,title=f'hmm {x+1}')
    return plt.show()



def show_year_diffs(dwm,hmm,years, cmap='gray'):
    """
    Function to plot change between years

    Args:
        dwm (dict): Dynamic World Model outputs, with year ints as keys and rasterio dataset pointers as values
        hmm (dict): Hidden Markov Model outputs, with year ints as keys and rasterio dataset pointers as values
        years (list): years to loop through for calculating annual change
        cmap (cmap): colormap

    Returns:
        Plots annual change layer for each year
    """
    fig, axs = plt.subplots(2,len(years)-1,figsize=((len(years)-1)*6,2*6))
    for i,x in enumerate(years[:-1]):
        show(np.equal(dwm[f'{x}'].read().argmax(axis=0),dwm[f'{x+1}'].read().argmax(axis=0)), 
             ax=axs[0,i], cmap=cmap, title=f'dwm {x}-{x+1}')
        show(np.equal(hmm[f'{x}'].read().argmax(axis=0),hmm[f'{x+1}'].read().argmax(axis=0)), 
             ax=axs[1,i], cmap=cmap,title=f'hmm {x}-{x+1}')
    return plt.show()

def show_max_probas(dwm,hmm,years, cmap=cmocean.cm.rain):
    """
    Function to plot max probability across bands for each year

    Args:
        dwm (dict): Dynamic World Model outputs, with year ints as keys and rasterio dataset pointers as values
        hmm (dict): Hidden Markov Model outputs, with year ints as keys and rasterio dataset pointers as values
        years (list): years to loop through for calculating annual change
        cmap (cmap): colormap

    Returns:
        Plots max probability per cell for each year
    """
    fig, axs = plt.subplots(len(years),2,figsize=(2*4,len(years)*4))
    for i,x in enumerate(years):
        show(dwm[f'{x}'].read().max(axis=0), ax=axs[i,0], cmap=cmap, vmin=0, vmax=1,title=x)
        show(hmm[f'{x}'].read().max(axis=0), ax=axs[i,1], cmap=cmap, vmin=0, vmax=1,title=x)
    return plt.show()


def show_normalized_diff(hmm_outputs, dwm_outputs, year, cmap=cmocean.cm.balance, band_names=DW_CLASS_LIST):
    """
    Function produce normalized difference plots for probability bands

    Args:
        hmm_outputs (dict): a in (a-b)/(a+b)
        dwm_outputs (dict): b in (a-b)/(a+b)
        year (int or str): year selection. Should be a key in both `hmm_outputs` and `dwm_outputs`
        cmap (cmap): valid colormap, should be diverging
        band_names (list): list of band names (str) to pass to plot titles

    Returns:
        Displays grid of plots showing normalized difference in 
    """
    fig, axs = plt.subplots(dwm_outputs[f'{year}'].count//3,3,figsize=(16,16))
    for i,x in enumerate(band_names):
        band=i+1
        a = hmm_outputs[f'{year}'].read(band)
        b = dwm_outputs[f'{year}'].read(band)
        show(np.divide(a-b,a+b+1e-8), ax=axs[(i//3),(i%3)], cmap=cmap, vmin=-1, vmax=1,title=x)
    return plt.show()

def show_label_agreement(dwm, hmm, label, year, cmap='gray'):
    """
    Function plot a visual comparison of the models and groud truth labels

    Args:
        dwm (dict): Dynamic World Model outputs, with year ints as keys and rasterio dataset pointers as values
        hmm (dict): Hidden Markov Model outputs, with year ints as keys and rasterio dataset pointers as values
        label (rio.Dataset): rasterio dataset pointer to the label object
        year (int): year to select for hmm and dwm
        cmap (str): colormap

    Returns:
        Plots DWM and HMM model labels (argmax) alongside ground truth label, with a difference layer
    """
    fig, axs = plt.subplots(1,5,figsize=(5*5,1*5))
    show(np.equal(dwm[f'{year}'].read().argmax(axis=0)+1,label.read()),alpha=label.dataset_mask()/255,
         ax=axs[0], cmap=cmap, title=f'dwm-label diff')
    show(dwm[f'{year}'].read().argmax(axis=0)+1,ax=axs[1],alpha=label.dataset_mask()/255,
         cmap=dw_class_cmap,norm=dw_norm,title=f'dwm class')
    show(label,ax=axs[2],cmap=dw_class_cmap,norm=dw_norm,title=f'label')
    show(hmm[f'{year}'].read().argmax(axis=0)+1,ax=axs[3],alpha=label.dataset_mask()/255,
             cmap=dw_class_cmap,norm=dw_norm,title=f'hmm class')
    show(np.equal(hmm[f'{year}'].read().argmax(axis=0)+1,label.read()),alpha=label.dataset_mask()/255, 
             ax=axs[4], cmap=cmap,title=f'hmm-label diff'),
    return plt.show()

def show_label_confidence(model, year, cmap='gray_r'):
    """
    Function to plot first and second choice classes (argmax), probability/confidence, and a composite of class label and confidence

    Args:
        model (dict): year ints as keys and rasterio dataset pointers as values
        year (int): year
        cmap: colormap

    Returns:
        Plots first and second choice labels and associated probability/confidence, along with a composite of the two. 
    """
    fig, axs = plt.subplots(2,3,figsize=(3*6,2*6))
    show(np.argsort(model[f'{year}'].read(),axis=0)[-1]+1,cmap=dw_class_cmap,norm=dw_norm, title=f'Most likely label',ax=axs[0,0])
    show(np.sort(model[f'{year}'].read(),axis=0)[-1],cmap=cmap,vmin=0,vmax=1,ax=axs[0,1],title='argmax')
    show(np.argsort(model[f'{year}'].read(),axis=0)[-1]+1,alpha=np.sort(model[f'{year}'].read(),
        axis=0)[-1],cmap=dw_class_cmap,norm=dw_norm,ax=axs[0,2],title='labels with conf as alpha')
    
    show(np.argsort(model[f'{year}'].read(),axis=0)[-2]+1,cmap=dw_class_cmap,norm=dw_norm, title=f'2nd most likely label',ax=axs[1,0])
    show(np.sort(model[f'{year}'].read(),axis=0)[-2],cmap=cmap,vmin=0,vmax=1,ax=axs[1,1],title='argmax (-1)')
    show(np.argsort(model[f'{year}'].read(),axis=0)[-2]+1,alpha=np.sort(model[f'{year}'].read(),
        axis=0)[-2],cmap=dw_class_cmap,norm=dw_norm,ax=axs[1,2],title='labels with conf as alpha')

    return plt.show()

def number_of_class_changes(model, years):
    """
    Function to count class changes

    Args:
        model (dict): year ints as keys and rasterio dataset pointers as values
        years (list): years to loop through for calculating annual change

    Returns:
        Array of class cumulative class changes per pixel
    """
    array = np.zeros_like(model[f'{years[0]}'].read().argmax(axis=0))
    for x in years[:-1]:
        array += np.not_equal(model[f'{x}'].read().argmax(axis=0),model[f'{x+1}'].read().argmax(axis=0)).astype(int)
    return array

def calc_ee_class_transitions(img1, img2, method='frequency_hist',scale=10,numPoints=500):
    """
    Function to calculate class transitions between two ee.image.Image objects.

    Args:
        img1 (ee.image.Image): first imput image (from/before)
        img2 (ee.image.Image): second imput image (to/after)
        method (str): `frequency_hist` or `stratified_sample`. Frequency_hist approach reduces the image stack 
            to output a frequency histogram defining class transitions. Stratified_sample retrieves a stratified sample of points
            and the built in error_matrix method. 
        scale (int): scale in meters, defaults to 10
        numPoints (int): number of points in stratified sample method, defaults to 500

    Returns:
        pd.DataFrame of class transition counts from img1 to img2 
    """
    if not all(isinstance(i, ee.image.Image) for i in [img1, img2]):
        print('inputs should be type ee.image.Image, YMMV')
    df = pd.DataFrame()
    if method == 'frequency_hist':
        hist = img1.addBands(img2).reduceRegion(
            reducer=ee.Reducer.frequencyHistogram().group(groupField=1,groupName='class'),
            bestEffort=True,
            scale=scale)
        hist_table = hist.getInfo()
        df_tm = pd.json_normalize(hist_table['groups']).set_index('class').fillna(0)
        df = df.add(df_tm, fill_value=0)
        df.index.name = None
        df.columns = [x.replace('histogram.','') for x in df.columns] # remove the 'histogram.' prefix from json io parse
        cols = sorted(df.columns.values,key=int) # sort string numbers by pretending they're real numbers
        df = df[cols].fillna(0) # nans are actually 0 in this case
        return df.T
    if method == 'stratified_sample':
        stacked_classes = img1.rename('before').addBands(img2.rename('after'))
        samples = stacked_classes.stratifiedSample(classBand='before',numPoints=numPoints,scale=scale)
        trns = samples.errorMatrix(actual='before',predicted='after')
        transition_mtx = trns.getInfo()
        return pd.DataFrame(transition_mtx)
    
def transition_matrix(df, kind='classwise', style=True, fmt='{:.2f}', cmap='BuPu'):
    """
    Function to normalize and style a confusion matrix DataFrame

    Args:
        df (pd.DataFrame): DataFrame of confusion matrix counts
        kind (str): type of normalization
            'changes_only': % of all class changes, exclusing those remaining the same
            'classwise_changes_only': % of all class changes by row, excluding those remaining the same
            'overall': % of all transitions including classes remaining (non-changes, identity matrix)
            'classwise':  % of all transitions by row including classes remaining (non-changes, identitity matrix)
        stye (bool): defaults True, whether to show style modifiers on output DataFrame
        fmt (str): number formatter pattern
        cmap (str): colormap to pass to DataFrame background_gradient property

    Returns:
        pd.DataFrame of normalized confusion matrix    
    """
    g=df.copy()
    if kind=='changes_only':
        np.fill_diagonal(g.values, 0)
        g = g.divide(sum(g.sum(axis=1)),axis=0)
    if kind=='classwise_changes_only':
        np.fill_diagonal(g.values, 0)
        g = g.divide(g.sum(axis=1),axis=0)
    if kind=='overall':
        g = g.divide(sum(g.sum(axis=1)),axis=0)
    if kind=='classwise':
        g = g.divide(g.sum(axis=1),axis=0)
    if style:
        return g.style.format(fmt).background_gradient(cmap=cmap,axis=None)
    if not style:
        return g


def all_the_stats(array):
    """
    Function calculate ancillary classification metrics

    Args:
        array (np.array): should be a multi-class confusion matrix

    Returns:
        Pandas.DataFrame of additional metrics by row

    Special reguards to https://en.wikipedia.org/wiki/Confusion_matrix
    """
    fp = np.clip((array.sum(axis=0) - np.diag(array)),1e-8,np.inf).astype(float)
    tp = np.clip(np.diag(array),1e-8,np.inf).astype(float)
    fn = np.clip((array.sum(axis=1) - np.diag(array)),1e-8,np.inf).astype(float)
    tn = np.clip((array.sum() - (fp + fn + tp)),1e-8,np.inf).astype(float)
    df = pd.DataFrame({
        'FP':fp, #false positive
        'TP':tp, #true positive
        'FN':fn, #false negative
        'TN':tn, #true negative
        'TPR':tp/(tp+fn), #true positive rate (sensitivity, hit rate, recall)
        'TNR':tn/(tn+fp), #true negative rate (specificity)
        'PPV':tp/(tp+fp), #positive predictive value (precision)
        'NPV':tn/(tn+fn), #negative predictive value
        'FPR':fp/(fp+tn), #false positive rate (fall out)
        'FNR':fn/(tp+fn), #false negative rate
        'FDR':fp/(tp+fp), #false discovery rate
        'ACC':(tp+tn)/(tp+fp+fn+tn), #overall class accuracy
    })
    return df