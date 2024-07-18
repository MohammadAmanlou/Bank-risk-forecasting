# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:49:33 2024

@author: M.Amanlou
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import seaborn as sns
   
class DataExplanation:
    def __init__(self , data, label):
        self.data = data.data
        self.label = label
        self.features = data.getColumns()
        self.numericalFeatures = list(self.data.select_dtypes(include = ["int" , "float"]).columns)
        self.categoricalFeatures = list(self.data.select_dtypes(include = ["object"]).columns)
        for feature in self.numericalFeatures :
            if len(self.data[feature].unique()) <= 10:
                self.categoricalFeatures.append(feature)
                self.numericalFeatures.remove(feature)
     
    def correlation(self):
        cor =  self.data[self.numericalFeatures].corr(numeric_only=True)
        plt.figure(figsize=(15,15))
        sns.heatmap(cor, annot=True, center=0 , fmt=".3f" , cmap = "rocket" , linewidth=0.75)
        plt.title('Correlation')
        plt.show()
        return cor
    
    def scatterPlot(self):
        fig, axes = plt.subplots( nrows=len(self.numericalFeatures), ncols=1, figsize=(7, 4*len(self.numericalFeatures)))
        for i, col in enumerate(self.numericalFeatures):

            axes[i].scatter( self.data[col] , self.data[self.label] , c="blue", alpha=0.5, marker=r'$\clubsuit$',
                   label="Luck")
            axes[i].set_title(col + ' to ' + self.label + " relation")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(self.label)
        plt.tight_layout()
        plt.show()
        
    def uniqueVals(self):
        nu = self.data.nunique().reset_index()
        plt.figure(figsize=(10,5))
        plt.title("number of unique vals for every f ")
        plt.xticks(rotation = 80)
        nu.columns = ['feature','num of unique vals']
        ax = sns.barplot(x='feature', y='num of unique vals', data=nu)
        return nu
    
    def distribution(self):
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(self.numericalFeatures, 1):
            plt.subplot(5, 5, i)
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
        
    def category_count(self):
        category_counts = {}
        for col in self.categoricalFeatures:
            category_counts[col] = self.data[col].value_counts()
            if(len(self.data[col].value_counts())<50):
                plt.figure(figsize=(5, 3))
                category_counts[col].plot(kind='bar', color='skyblue')
                plt.title(f'Counts of Data in Unique Categories for {col}')
                plt.xlabel(f'{col}')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.show()

        for col, counts in category_counts.items():
            if(len(counts)<10):
                print(f"Category counts for column '{col}':")
                print(counts)
                print()
                
    def monteCarlo_simulation(self, feature, num_simulations = 1000):
        simulated = []
        for _ in range(num_simulations):
            # Generating random samples for analysis
            simulated.append(np.random.normal(self.data[feature].mean(), self.data[feature].std()))

        simulated_prices_mean = np.mean(simulated)
        simulated_prices_std = np.std(simulated)

        print(f'Mean simulated price: {simulated_prices_mean}')
        print(f'Standard deviation of simulated price: {simulated_prices_std}')

    def central_limitt(self, confidence_level = 0.95):
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        confidence_intervals = {}

        for col in self.data.select_dtypes(include=[np.number]).columns:
            sample = self.data[col]
            
            sample_mean = sample.mean()
            sample_std = sample.std(ddof=1)
            standard_error = sample_std / np.sqrt(len(sample))
            
            margin_of_error = z_score * standard_error
            
            confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
            
            confidence_intervals[col] = confidence_interval

        for col, interval in confidence_intervals.items():
            print(f"95% Confidence Interval for {col}: {interval}")
            
    def violon_plot(self, feature):
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.data, x=feature)
        plt.title(feature + ' Distribution')
        plt.xlabel(feature)
        plt.show()
        
    def box_plot(self):
        for column in self.numericalFeatures:
            sns.boxplot(x=self.data[column])
            plt.title(f'Box Plot of {column}')
            plt.show()