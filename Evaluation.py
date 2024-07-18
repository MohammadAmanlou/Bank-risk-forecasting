from utils import *
from dataset import *
from preprocess import *
from data import *
from Models import *
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

#%%
def get_rmse_score(preds , actual):
    if len(preds) != len(actual):
        raise ValueError("The lengths of predicted and actual values must match.")
    residuals = np.subtract(preds, actual)
    mean_squared_error = np.mean(np.square(residuals))
    return np.sqrt(mean_squared_error)

def get_r2_score(predicted_values, actual_values):
    if len(predicted_values) != len(actual_values):
        raise ValueError("The lengths of predicted and actual values must match.")

    mean_actual_values = sum(actual_values) / len(actual_values)
    total_sum_of_squares = sum((y_i - mean_actual_values) ** 2 for y_i in actual_values)
    residual_sum_of_squares = sum((y_i - y_hat) ** 2 for y_i, y_hat in zip(actual_values, predicted_values))
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared

class Evaluation:
    def __init__(self , output , preprocess):
        self.preds = output['pred']
        self.fit = output['fit']
        self.actualFit = preprocess.y_train
        self.actualPreds = preprocess.y_test
    
    def get_test_rmse_score(self):
        return get_rmse_score(self.preds , self.actualPreds)
    def get_train_rmse_score(self):
        return get_rmse_score(self.fit , self.actualFit)
    def get_test_r2_score(self):
        return get_r2_score(self.preds , self.actualPreds)
    def get_train_r2_score(self):
        return get_r2_score(self.fit , self.actualFit)
    
    def get_data_percentage_in_dist_train(self , dist):
        c = 0
        for i in list(self.fit - self.actualFit):
            if (i  <= dist and i  >= -1 * dist):
                c += 1
        return (c / len(self.fit))/2 + 0.5
    
    def get_data_percentage_in_dist_test(self , dist):
        c = 0
        for i in list(self.preds - self.actualPreds):
            if (i  <= dist and i  >= -1 * dist):
                c += 1
        return (c / len(self.preds))/2 + 0.5
        
        
        
        
        
# %%
