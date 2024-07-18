# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:39:32 2024

@author: M.Amanlou
"""


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from utils import *
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None

class Preprocess():
    def __init__(self  , data , label):
        self.data = data.data
        self.features = data.getColumns()
        self.numericalFeatures = []
        self.categoricalFeatures = []
        self.label = label
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def apply_polynomial_features(self, degree):
        poly_features = PolynomialFeatures(degree=degree)
        self.X_train = poly_features.fit_transform(self.X_train)
        self.X_test = poly_features.transform(self.X_test)
    
    def makeClassificationLabel (self , prevMonthLabel):
        self.data[self.label] = (self.data[self.label] - self.data[prevMonthLabel] ) / self.data[prevMonthLabel] *100
        self.data["temp"] = 7
        self.data['temp'][self.data[self.label] < -50]  = 1
        self.data['temp'][(self.data[self.label] >= -50) & (self.data[self.label] < -30)]  = 2
        self.data['temp'][(self.data[self.label] >= -30) & (self.data[self.label] < -5)]  = 3
        self.data['temp'][(self.data[self.label] >= -5) & (self.data[self.label] < 5)]  = 4
        self.data['temp'][(self.data[self.label] >= 5) & (self.data[self.label] < 30)]  = 5
        self.data['temp'][(self.data[self.label] >= 30) & (self.data[self.label] < 50)]  = 6
        self.data[self.label] = self.data["temp"]
        self.data.drop(columns = ["temp"] , inplace = True)
        
    def fillna(self, val):
        self.data.fillna(val , inplace = True)
        
    def trainTestSplit(self , test_size):
        if (len(self.data[self.label].unique()) <= 10):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.data.drop(columns=[self.label]),  # features
                self.data[self.label],                 # target
                test_size=test_size,               # 75/25 split, a quarter of data used for testing
                random_state= 42              # Seed for random number generator for reproducibility
                ,stratify=self.data[self.label]         # Stratify split based on labels, if classification problem
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.data.drop(columns=[self.label]),  # features
                self.data[self.label],                 # target
                test_size=test_size,               # 75/25 split, a quarter of data used for testing
                random_state= 42 )             # Seed for random number generator for reproducibility
        self.X_test = self.X_test.reset_index().drop("index" , axis =1 )    
        self.X_train = self.X_train.reset_index().drop("index" , axis =1 )    
    
            
    def divideNumericalCategorical(self):
        self.numericalFeatures = list(self.data.select_dtypes(include = ["int" , "float"]).columns)
        self.categoricalFeatures = list(self.data.select_dtypes(include = ["object"]).columns)
        for feature in self.numericalFeatures :
            if len(self.data[feature].unique()) <= 10:
                self.categoricalFeatures.append(feature)
                self.numericalFeatures.remove(feature)
        if (self.label in self.numericalFeatures):
            self.numericalFeatures.remove(self.label)
        else:
            self.categoricalFeatures.remove(self.label)
                
    def minMaxScale(self ):
        scaler = MinMaxScaler()
        return scaler
    
    def standardScale(self):
        scaler = StandardScaler()
        return scaler
    
    def runTrainScale(self , scaler):
        temp = pd.DataFrame(scaler.fit_transform(self.X_train[self.numericalFeatures]), columns=self.numericalFeatures, index=self.X_train.index) 
        data = pd.concat([temp ,self.X_train[self.categoricalFeatures] ] , axis = 1 )
        self.X_train = data
        # temp = pd.DataFrame((self.y_train - self.y_train.min()) /  (self.y_train.max() - self.y_train.min()))
        # self.y_train = temp
        return data,self.y_train
    
    def runTestScale(self , scaler):
        temp = pd.DataFrame(self.X_test[self.numericalFeatures], columns=self.numericalFeatures, index=self.X_test.index).copy()
        for col in self.numericalFeatures:
            if ((self.X_train[col].max() - self.X_train[col].min()) != 0):
                temp[col] = (self.X_test[col] - self.X_train[col].min()) /  (self.X_train[col].max() - self.X_train[col].min())
        X_test = pd.concat([temp , self.X_test[self.categoricalFeatures] ] , axis = 1 )
        #temp = pd.DataFrame((self.y_test - self.y_train.min()) /  (self.y_train.max() - self.y_train.min()))
        self.X_test = X_test
        #self.y_test = temp
        return X_test,self.y_test

    def outlierDroppingQuantileTrain(self , inplace, quantile):
        self.X_train["outlier"] = 0
        self.X_train[self.label] = 0
        self.X_train[self.label] = self.y_train.values # Assign values of self.y_train 
        for col in self.numericalFeatures:
            self.X_train["outlier"][(self.X_train[col] > self.X_train[col].quantile(quantile) )] = 1
    
        if (inplace == True):
            self.X_train = self.X_train[self.X_train["outlier"] == 0]
            self.X_train.reset_index(drop=True, inplace=True)  # Reset index after filtering
            self.y_train = np.zeros(len(self.X_train[self.label]))
            self.y_train = self.X_train[self.label]
            self.X_train.drop(columns=["outlier", self.label], inplace=True)
        else:
            return self.X_train[self.X_train["outlier"] == 0].drop(columns = ["outlier", self.label] )
        
    def outlierDroppingQuantileTest(self , inplace):
        self.X_test["outlier"] = 0 
        self.X_test[self.label] = self.y_test.values
        self.X_train[self.label] = self.y_train.values
        for col in self.numericalFeatures: 
            self.X_test["outlier"][(self.X_test[col] > self.X_train[col].max() )] = 1
             
        if (inplace == True):
            self.X_test = self.X_test[self.X_test["outlier"] == 0]
            self.y_test = self.X_test[self.label]
            self.X_test.reset_index(drop=True, inplace=True)  # Reset index after filtering
            self.X_test.drop(columns = ["outlier" , self.label] , inplace = True)
            self.X_train.drop(columns = [self.label] , inplace = True)
        else:
            temp = self.X_test[self.X_test["outlier"] == 0].drop(columns = ["outlier", self.label] )
            self.X_test.drop(columns = ["outlier" , self.label] , inplace = True)
            self.X_train.drop(columns = [self.label] , inplace = True)
            return temp
    
    def outlier_dropping_ZScore_train(self):
        threshold_values = range(1, 11)
        outliers_removed = []
        for threshold in threshold_values:
            df_cleaned = drop_numerical_outliers(self.X_train, self.numericalFeatures, threshold)
            num_outliers_removed = len(self.X_train) - len(df_cleaned)
            outliers_removed.append(num_outliers_removed)
        # Plot the number of outliers removed for each threshold value
        plt.plot(threshold_values, outliers_removed)
        plt.xlabel('Threshold Value')
        plt.ylabel('Number of Outliers Removed')
        plt.title('Effect of Different Threshold Values on Outlier Removal')
        plt.show()
    
    def outlier_dropping_ZScore_test(self):
        threshold_values = range(1, 11)
        outliers_removed = []
        for threshold in threshold_values:
            df_cleaned = drop_numerical_outliers(self.X_test, self.numericalFeatures, threshold)
            num_outliers_removed = len(self.X_test) - len(df_cleaned)
            outliers_removed.append(num_outliers_removed)
        # Plot the number of outliers removed for each threshold value
        plt.plot(threshold_values, outliers_removed)
        plt.xlabel('Threshold Value')
        plt.ylabel('Number of Outliers Removed')
        plt.title('Effect of Different Threshold Values on Outlier Removal')
        plt.show()
    
    
    def nullHandling(self, method, column , **kwargs):
        if column not in self.data.columns:
            print("Invalid column specified")
            return self.data

        match method:
            case "fill":
                value = kwargs.get('value', 0)  # Default fill value is 0
                self.data[column] = self.data[column].fillna(value)
                return self.data

            case "interpolate":
                interp_method = kwargs.get('method', 'linear')  # Default interpolation method is linear
                limit_direction = kwargs.get('limit_direction', 'forward')
                self.data[column] = self.data[column].interpolate(method=interp_method, limit_direction=limit_direction)
                return self.data

            case "random":
                if self.data[column].isnull().any():
                    random_values = np.random.choice(self.data[column].dropna().values, size=self.data[column].isnull().sum())
                    self.data.loc[self.data[column].isnull(), column] = random_values
                return self.data

            case "imputers":
                strategy = kwargs.get('strategy', 'mean')  # Default strategy is mean
                if strategy == 'mean' or strategy == 'median' or strategy == 'most_frequent' or strategy == 'constant':
                    imputer = SimpleImputer(strategy=strategy)
                    self.data[column] = imputer.fit_transform(self.data[[column]])
                elif strategy == 'knn':
                    n_neighbors = kwargs.get('n_neighbors', 5)  # Default number of neighbors is 5
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    self.data[column] = imputer.fit_transform(self.data[[column]])
                elif strategy == 'random_forest':
                    self.data = impute_with_random_forest(self.data, column)
                elif strategy == 'linear_regression':
                    self.data = impute_with_linear_regression(self.data, column)
                return self.data

            case "drop":
                self.data = self.data[column].dropna()
                return self.data

            case _:
                print("Invalid method specified")
                return self.data
        
    def drop_null_label_rows(self):
        self.data = self.data.dropna(subset=[self.label])
    
    def preprocess(self , testSize, prevMonthLabel ,scaler = "MinMax" , outlierQuantile = 0.99, fillnaVal = 0 , classifyLabel = "classifyLabel" ):
        self.drop_null_label_rows()
        for col in self.data.columns:
            self.nullHandling("fill", col, value = fillnaVal)
        self.divideNumericalCategorical()
        self.trainTestSplit(testSize)
        self.outlierDroppingQuantileTrain(inplace = True , quantile = outlierQuantile)
        self.outlierDroppingQuantileTest(inplace = True)

        if(scaler == "MinMax"):
           scaler_ = self.minMaxScale()
        elif(scaler == "Standard"):
            scaler_ = self.standardScale()
        X_testScaled , y_testScaled = self.runTestScale(scaler_)
        X_trainScaled , y_trainScaled = self.runTrainScale(scaler_)
        return X_trainScaled,y_trainScaled,X_testScaled,y_testScaled
    
    def classificationPreprocess(self , testSize, prevMonthLabel ,scaler = "MinMax" , outlierQuantile = 0.99, fillnaVal = 0  ):
        self.makeClassificationLabel (prevMonthLabel)
        self.drop_null_label_rows()
        for col in self.data.columns:
            self.nullHandling("fill", col, value = fillnaVal)
        self.divideNumericalCategorical()
        self.trainTestSplit(testSize)
        self.outlierDroppingQuantileTrain(inplace = True , quantile = outlierQuantile)
        self.outlierDroppingQuantileTest(inplace = True)
        if(scaler == "MinMax"):
           scaler_ = self.minMaxScale()
        elif(scaler == "Standard"):
            scaler_ = self.standardScale()
        X_testScaled , y_testScaled = self.runTestScale(scaler_)
        X_trainScaled , y_trainScaled = self.runTrainScale(scaler_)
        return X_trainScaled,y_trainScaled,X_testScaled,y_testScaled

    def __len__(self):
        return len(self.X_train)
    
    def len(self):
        return len(self.X_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]