# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:46:41 2024

@author: 2281580024
"""

#%% import libs

from utils import *
import shap
import os
import subprocess
import time
import config
from config import Config

from sklearn import svm as sk_svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import tree
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_score, silhouette_score


import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

#%%

class Data:
    def __init__(self , data):
        self.data = data
        self.uniqueVals = {}
    
    def addUniqueVals(self , key , value):
        self.uniqueVals[key] = value
        
    def getColumns(self):
        return self.data.columns
        
        
class Dataset:
    def __init__ (self ,fileName , spliters):
        self.fileName = fileName
        self.allData = pd.read_csv( "C://Users//a//Desktop//DAATA//" + self.fileName )
        self.spliters = spliters
        self.dividedDataset = []
    
    def divideByFeatures (self  ):
        toDivedDatasets = [self.allData]
        for feature in  self.spliters:
            toDivedDatasets = toDivedDatasets + self.dividedDataset
            self.dividedDataset = []
            uniqueVals = self.allData[feature].unique()
            for value in uniqueVals:
                for dataset in toDivedDatasets:
                    data = Data(dataset[dataset[feature] == value])
                    data.addUniqueVals(feature, value)
                    self.dividedDataset.append(data)
                toDivedDatasets = []
                
    
    def getDividedDatasets(self):
        self.divideByFeatures()
        return self.dividedDataset
    