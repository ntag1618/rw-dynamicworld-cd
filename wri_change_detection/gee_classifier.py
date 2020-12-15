#Import necessary libraries
import os
import ee
import numpy as np
import pandas as pd
import random
import itertools

def pretty_print_confusion_matrix(confusion_list):
    """
    Function to print a confusion matrix list

    Args:
        confusion_list (List): a list of confusion matrix values, can be taken from ee.ConfusionMatrix().getInfo()

    Returns:
        An pandas.DataFrame of confusion matrix with column names and row names
    """
    out_confusion_matrix = pd.DataFrame({'_':['Observed_False','Observed_True'],
                                 'Predicted_False':confusion_list[:][0],
                                 'Predicted_True':confusion_list[:][1]})

    out_confusion_matrix = out_confusion_matrix.set_index('_')
    return out_confusion_matrix

def buildGridSearchList(parameters,classifier_name):
    """
    Function to build a list of classifiers to use in kFoldCrossValidation to test mutliple parameters similar to scikit learn's gridSearchCV

    Args:
        parameters (Dictionary): dictionary of parameters from ee.Classifier
        classifier_name (String): name of the classifier as a string in the last part of ee.Classifier.classifier_name

    Returns:
        A list of dictionaries of classifiers and parameters to be used in kFoldCrossValidation
    """
    param_keys, param_values = zip(*parameters.items())
    param_list = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]
    
    classifier_list = None
    if classifier_name == 'smileRandomForest':
        classifier_list = [{'Type':'smileRandomForest','Params':str(x),'Classifier':ee.Classifier.smileRandomForest(**x)} for x in param_list]
    elif classifier_name == 'smileNaiveBayes':
        classifier_list = [{'Type':'smileNaiveBayes','Params':str(x),'Classifier':ee.Classifier.smileNaiveBayes(**x)} for x in param_list]
    elif classifier_name == 'libsvm':
        classifier_list = [{'Type':'libsvm','Params':str(x),'Classifier':ee.Classifier.libsvm(**x)} for x in param_list]
    elif classifier_name == 'gmoMaxEnt':
        classifier_list = [{'Type':'gmoMaxEnt','Params':str(x),'Classifier':ee.Classifier.gmoMaxEnt(**x)} for x in param_list]
    return classifier_list
    
    
def defineClassifier(parameters,classifier_name):
    """
    Function to take parameters and classifier_name and load ee.Classifier

    Args:
        parameters (Dictionary): dictionary of parameters from ee.Classifier
        classifier_name (String): name of the classifier as a string in the last part of ee.Classifier.classifier_name

    Returns:
        An ee.Classifier object with inputted parameters
    """
    classifier = None
    if classifier_name == 'smileRandomForest':
        classifier = ee.Classifier.smileRandomForest(**parameters)
    elif classifier_name == 'smileNaiveBayes':
        classifier = ee.Classifier.smileNaiveBayes(**parameters)
    elif classifier_name == 'libsvm':
        classifier = ee.Classifier.libsvm(**parameters)
    elif classifier_name == 'gmoMaxEnt':
        classifier = ee.Classifier.gmoMaxEnt(**parameters)
    else:
        print('Classifier not recognized')
    return classifier
    

def kFoldCrossValidation(inputtedFeatureCollection, propertyToPredictAsString, predictors, listOfClassifiers, k, seed=200):
    """
    Args:
    inputtedFeatureCollection (ee.FeatureCollection): an ee.FeatureCollection() of sample points object with a property of interest
    propertyToPredictAsString (String): the property to predict
    predictors (List of Strings): properties to use in training
    listOfClassifiers (List of dictionaries): a list of classifiers created using buildGridSearchList
    k (Int): the number of folds
    seed (Int): seed to use in training

    Returns:
        An ee.Feature Collection of cross validation results, with training and validaiton score, parameters, and classifier name
    
    Much of this code was taken from Devin Routh's [https://devinrouth.com/] work at the Crowther Lab at ETH Zurich [https://www.crowtherlab.com/]
    The code is released under Apache License Version 2.0 [http://www.apache.org/licenses/], and you can learn more about the license here [https://gitlab.ethz.ch/devinrouth/crowther_lab_nematodes/-/blob/master/LICENSE]
    The code was originally written in JavaScript and was converted to Python and adapted for our purposes by Kristine Lister.
    You can find the original code written by Devin in this Earth Engine toolbox: users/devinrouth/toolbox:KFoldCrossValConvolveGapFillEnsemble.js
    """
    np.random.seed(seed)
    #The sections below are the function's code, beginning with
    #preparation of the inputted feature collection of sample points

    collLength = inputtedFeatureCollection.size()
    print('Number of Sample Points',collLength.getInfo())

    sampleSeq = ee.List.sequence(1, collLength)

    inputtedFCWithRand = inputtedFeatureCollection.randomColumn('Rand_Num', seed).sort('Rand_Num').toList(collLength)

    # Prep the feature collection with random fold assignment numbers
    preppedListOfFeats = sampleSeq.map(lambda numberToSet: ee.Feature(inputtedFCWithRand.get(ee.Number(numberToSet).subtract(1))).set('Fold_ID', ee.Number(numberToSet)))
    
    # ———————————————————————————————————————————————————————————————
    # This section divides the feature collection into the k folds

    averageFoldSize = collLength.divide(k).floor()
    print('Average Fold Size',averageFoldSize.getInfo())

    remainingSampleSize = collLength.mod(k)

    def fold_function(fold):
        foldStart = ee.Number(fold).multiply(averageFoldSize).add(1)
        foldEnd = ee.Number(foldStart).add(averageFoldSize.subtract(1))
        foldNumbers = ee.List.sequence(foldStart, foldEnd)
        return ee.List(foldNumbers)

    foldSequenceWithoutRemainder = ee.List.sequence(0, k - 1).map(fold_function)

    remainingFoldSequence = ee.List.sequence(ee.Number(ee.List(foldSequenceWithoutRemainder.get(foldSequenceWithoutRemainder.length().subtract(1))).get(averageFoldSize.subtract(1))).add(1),
        ee.Number(ee.List(foldSequenceWithoutRemainder.get(foldSequenceWithoutRemainder.length().subtract(1))).get(averageFoldSize.subtract(1))).add(ee.Number(remainingSampleSize)))

    # This is a list of lists describing which features will go into each fold
    listsWithRemaindersAdded = foldSequenceWithoutRemainder.zip(remainingFoldSequence).map(lambda x: ee.List(x).flatten())

    finalFoldLists = listsWithRemaindersAdded.cat(foldSequenceWithoutRemainder.slice(listsWithRemaindersAdded.length()))

    mainFoldList = ee.List.sequence(0, k - 1)

    # Make a feature collection with a number of null features equal to the number of folds
    # This is done to stay in a collection rather than moving to a list
    foldFeatures = ee.FeatureCollection(mainFoldList.map(lambda foldNumber: ee.Feature(None).set({'Fold_Number': ee.Number(foldNumber)})))
    # print('Null FC',foldFeatures)

    def assign_fold_number(feature):
        featureNumbersInFold = finalFoldLists.get(ee.Feature(feature).get('Fold_Number'))
        featuresWithFoldNumbers = ee.FeatureCollection(preppedListOfFeats).filter(ee.Filter.inList('Fold_ID', featureNumbersInFold)).map(lambda f: f.set('Fold_Number', ee.Feature(feature).get('Fold_Number')))
        return featuresWithFoldNumbers

    # Use the null FC to filter and assign a fold number to each feature, then flatten it back to a collection
    featuresWithFoldAssignments = foldFeatures.map(assign_fold_number).flatten()

    # ———————————————————————————————————————————————————————————————
    # Train the data and retrieve the values at the sample points
    
    def grid_search(classifier_object):
        classifier = classifier_object.get('Classifier')
        def cross_val(foldFeature):
            trainingFold = featuresWithFoldAssignments.filterMetadata('Fold_Number', 'not_equals', ee.Number(foldFeature.get('Fold_Number')))
            validationFold = featuresWithFoldAssignments.filterMetadata('Fold_Number', 'equals', ee.Number(foldFeature.get('Fold_Number')))
            trained_classifier = classifier.train(features=trainingFold, classProperty=propertyToPredictAsString, inputProperties=predictors)
            trainAccuracy = trained_classifier.confusionMatrix().accuracy()
            validation_points_predicted = validationFold.classify(trained_classifier)
            validationAccuracy = validation_points_predicted.errorMatrix(propertyToPredictAsString, 'classification').accuracy()

            foldFeature = foldFeature.set({'Training Score':trainAccuracy})
            foldFeature = foldFeature.set({'Validation Score':validationAccuracy})
            return foldFeature

        cross_val_results = foldFeatures.map(cross_val)
        average_training_score = cross_val_results.aggregate_mean('Training Score')
        average_validation_score = cross_val_results.aggregate_mean('Validation Score')
        
        classifier_feature = ee.Feature(ee.Geometry.Point([0,0])).set('Classifier Type',classifier_object.get('Type'))
        classifier_feature = classifier_feature.set('Params',classifier_object.get('Params'))
        classifier_feature = classifier_feature.set('CV Training Score',average_training_score)
        classifier_feature = classifier_feature.set('CV Validation Score',average_validation_score)
        return classifier_feature
        
    return ee.FeatureCollection([grid_search(x) for x in listOfClassifiers])
    

