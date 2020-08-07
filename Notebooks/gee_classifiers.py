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




def pretty_print_confusion_matrix(classifier):
    gee_error_matrix = np.array(classifier.confusionMatrix().getInfo())
    out_confusion_matrix = pd.DataFrame({'_':['Observed_False','Observed_True'],
                                 'Predicted_False':gee_error_matrix[:,0],
                                 'Predicted_True':gee_error_matrix[:,1]})

    out_confusion_matrix = out_confusion_matrix.set_index('_')
    return out_confusion_matrix
    
def train_rf_classifier(training_points, y_column, predictors, seed=None, numberOfTrees=100, 
                                        variablesPerSplit=None, minLeafPopulation=1, bagFraction=0.5, maxNodes=None):
    if seed:
        rf_classifier = ee.Classifier.smileRandomForest(
                                            seed=seed, 
                                            numberOfTrees=numberOfTrees, 
                                            variablesPerSplit=variablesPerSplit, 
                                            minLeafPopulation=minLeafPopulation, 
                                            bagFraction=bagFraction, 
                                            maxNodes=maxNodes)
        
    else:
        rf_classifier = ee.Classifier.smileRandomForest(
                                            numberOfTrees=numberOfTrees, 
                                            variablesPerSplit=variablesPerSplit, 
                                            minLeafPopulation=minLeafPopulation, 
                                            bagFraction=bagFraction, 
                                            maxNodes=maxNodes)
    rf_classifier = rf_classifier.train(features= training_points, classProperty= y_column, inputProperties= predictors)
    return rf_classifier
    
def train_naive_bayes(training_points, y_column, predictors, lambda_value=0.000001):
    naive_bayes = ee.Classifier.smileNaiveBayes(lambda_value)
    naive_bayes = naive_bayes.train(features= training_points, classProperty= y_column, inputProperties= predictors)
    return naive_bayes
    
def train_svm(training_points, y_column, predictors, decisionProcedure="Voting", svmType='C_SVC', kernelType='LINEAR', 
                                        shrinking=True, degree=None, gamma=None, coef0=None, cost=None, nu=None, 
                                        terminationEpsilon=None, lossEpsilon=None, oneClass=None):
    svm_classifier = ee.Classifier.libsvm(
                                        decisionProcedure=decisionProcedure,
                                        svmType=svmType,
                                        kernelType=kernelType,
                                        shrinking=shrinking,
                                        degree=degree,
                                        gamma=gamma,
                                        coef0=coef0,
                                        cost=cost,
                                        nu=nu,
                                        terminationEpsilon=terminationEpsilon,
                                        lossEpsilon=lossEpsilon,
                                        oneClass=oneClass
                                        )
    svm_classifier = svm_classifier.train(features= training_points, classProperty= y_column, inputProperties= predictors)
    return svm_classifier
    
def train_gmoMaxEnt(training_points, y_column, predictors, weight1=0, weight2= 0.000009999999747378752, 
                                                            epsilon= 0.000009999999747378752, minIterations=0, maxIterations=100):
    gmoMaxEnt_classifier = ee.Classifier.gmoMaxEnt(
                                            weight1=weight1, 
                                            weight2=weight2, 
                                            epsilon=epsilon, 
                                            minIterations=minIterations, 
                                            maxIterations=maxIterations)
    gmoMaxEnt_classifier = gmoMaxEnt_classifier.train(features= training_points, classProperty= y_column, inputProperties= predictors)
    return gmoMaxEnt_classifier
    












