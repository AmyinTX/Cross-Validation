# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:37:11 2017

@author: amybrown
"""

#%% import packages
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.cross_validation import train_test_split

#%% load and prepare data
iris = datasets.load_iris()

#split into testing and training sets using helper function
iris_train_feat, iris_test_feat, iris_train_target, iris_test_target = train_test_split(iris.data, iris.target, test_size=0.4, random_state=42)

#%% data is successfully split. now to fit a classifier on training set and evaluate on test data
model = svm.SVC(kernel='linear')
trainX = iris_train_feat
trainy = iris_train_target
model.fit(trainX, trainy)
model.score(trainX, trainy)

predicted = model.predict(iris_test_feat)

predicted == iris_test_target 
model.score(iris_test_feat, predicted) # [erfectly predicts the outputs


#%% second part: cross validation of SVC 
from sklearn.model_selection import cross_val_score
model = svm.SVC(kernel='linear')
scores = cross_val_score(model, iris.data, iris.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn import metrics
scores = cross_val_score(model, iris.data, iris.target, cv=5, scoring='f1_macro')

# how does accuracy scores compare to F1 scores?