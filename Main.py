
#%%
from utils import *
from dataset import *
from preprocess import *
from data import *
from Models import *
from Evaluation import *
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
from dataExplanation import *
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None
#%%
dataset = Dataset("Risk_Dataset_Haghighi.csv" , ["CUSTOMER_TP"])
data = dataset.getDividedDatasets()[0]
prp = Preprocess(data , "LABLE_JARI" )
prp.classificationPreprocess(0.2 , "SUMAVGAMNT_JARI")
config = Config()
dist = 1
classification = 1
#%%
de = DataExplanation(data ,"LABLE_JARI")
#%%
gaussianprocessOutput = gaussianprocess(prp , config , "guassianProcess")
gaussianprocessOutput['pred'] = np.argmax(gaussianprocessOutput['pred'], axis=1) + 1 
gaussianprocessOutput['fit'] = np.argmax(gaussianprocessOutput['fit'], axis=1) + 1
eval = Evaluation(gaussianprocessOutput , prp)
print_eval_metrics(eval, gaussianprocessOutput, dist, classification)
# %%
logregOutput = logreg(prp , config , 1.5 ,"logreg")
logregOutput['pred'] = np.argmax(logregOutput['pred'], axis=1) + 1 
logregOutput['fit'] = np.argmax(logregOutput['fit'], axis=1) + 1
eval = Evaluation(logregOutput , prp)
print_eval_metrics(eval, logregOutput, dist, classification)
# %%
extreeOutput = extree(prp, config, 1.5, 3, 1, "extree")
extreeOutput['pred'] = np.argmax(extreeOutput['pred'], axis=1) + 1 
extreeOutput['fit'] = np.argmax(extreeOutput['fit'], axis=1) + 1
eval = Evaluation(extreeOutput , prp)
print_eval_metrics(eval, extreeOutput, dist, classification)
# %%
forestOutput = forest(prp, config, 1.5, 3, 1, "forest")
forestOutput['pred'] = np.argmax(forestOutput['pred'], axis=1) + 1 
forestOutput['fit'] = np.argmax(forestOutput['fit'], axis=1) + 1
eval = Evaluation(forestOutput , prp)
print_eval_metrics(eval, forestOutput, dist, classification)
# %%
prp.y_train = prp.y_train - 1
prp.y_test = prp.y_test - 1
XGboostOutput = XGboost(prp, config, 1.5, 3, 1, "XGboost")
XGboostOutput['pred'] = np.argmax(XGboostOutput['pred'], axis=1) 
XGboostOutput['fit'] = np.argmax(XGboostOutput['fit'], axis=1) 
eval = Evaluation(XGboostOutput , prp)
print_eval_metrics(eval, XGboostOutput, dist, classification)
#%%
SVMSingleOutput = svm_single(prp, config, 1.5, 3, "SVMSingleOutput")
SVMSingleOutput['pred'] = np.argmax(SVMSingleOutput['pred'], axis=1) + 1 
SVMSingleOutput['fit'] = np.argmax(SVMSingleOutput['fit'], axis=1) + 1
eval = Evaluation(SVMSingleOutput , prp)
print_eval_metrics(eval, SVMSingleOutput, dist, classification)
#%%
SVMMultiOutput = svm_multi(prp, config, np.random.randint(0, 5, len(prp.y_train)),1.5, 3, "SVMMultiOutput")
SVMMultiOutput['pred'] = np.argmax(SVMMultiOutput['pred'], axis=1) + 1 
SVMMultiOutput['fit'] = np.argmax(SVMMultiOutput['fit'], axis=1) + 1
eval = Evaluation(SVMMultiOutput , prp)
print_eval_metrics(eval, SVMSingleOutput, dist, classification)
#%%
MLPSingleOutput = nnet_single(prp, 3 , config, "MLPSingleOutput")
MLPSingleOutput['pred'] = np.argmax(MLPSingleOutput['pred'], axis=1) + 1 
MLPSingleOutput['fit'] = np.argmax(MLPSingleOutput['fit'], axis=1) + 1
eval = Evaluation(MLPSingleOutput , prp)
print_eval_metrics(eval, MLPSingleOutput, dist, classification)
#%%
MLPMultiOutput = nnet_multi(prp, config, np.random.randint(0, 5, len(prp.y_train)), "MLPMultiOutput")
MLPMultiOutput['pred'] = np.argmax(MLPMultiOutput['pred'], axis=1) + 1 
MLPMultiOutput['fit'] = np.argmax(MLPMultiOutput['fit'], axis=1) + 1
eval = Evaluation(MLPMultiOutput , prp)
print_eval_metrics(eval, MLPMultiOutput, dist, classification)
###############################################################################################################
#%%
print("train RMSE score of LSTM: 41795836.66982718 \ntest RMSE score of LSTM: 64322550.95831925 \ntest R2 score of LSTM: 0.7290787246500784 \ntrain R2 score of LSTM: 0.7837978243455234")


# %%
dataset = Dataset("Risk_Dataset_Haghighi.csv" , ["CUSTOMER_TP"])
data = dataset.getDividedDatasets()[0]
prp = Preprocess(data , "LABLE_JARI" )
prp.preprocess(0.2 , "SUMAVGAMNT_JARI")
config = Config()
classification = 0
dist = 0 
#%%
forestRegressorOutput  = forest_regressor(prp, config, 1.5, 3, 0, "RFRegressor")
eval = Evaluation(forestRegressorOutput , prp)
print_eval_metrics(eval, forestRegressorOutput, dist, classification)
# %%
linearRegressorOutput  = linear_regressor(prp, config, 3, 0, "LSTM")
eval = Evaluation(linearRegressorOutput , prp)
print_eval_metrics(eval, linearRegressorOutput, dist, classification)
# %%
polynomialRegressorOutput = polynomial_regressor(prp, config, 3, 0, "polynomialRegressor", degree=5)
eval = Evaluation(polynomialRegressorOutput , prp)
print_eval_metrics(eval, polynomialRegressorOutput, dist, classification)

# %%
xgboostRegressorOutput = xgboost_regressor(prp, config,1.5, 3, 0, "xgboostRegressor")
eval = Evaluation(xgboostRegressorOutput , prp)
print_eval_metrics(eval, xgboostRegressorOutput, dist, classification)

# %%
KNNRegressorOutput = knn_regressor(prp, config,1.5, 3, 0, "KNNRegressor")
eval = Evaluation(KNNRegressorOutput , prp)
print_eval_metrics(eval, KNNRegressorOutput, dist, classification)

# %%
SVMSingleRegressorOutput = svm_single_reg(prp, 3 , config ,1.5, "SVMSingleRegressor")
eval = Evaluation(SVMSingleRegressorOutput , prp)
print_eval_metrics(eval, SVMSingleRegressorOutput, dist, classification)

# %%
SVMMultiRegressorOutput = svm_multi_reg(prp, config ,np.random.randint(0, 5, len(prp.y_train)),1.5, "SVMMultiRegressor")
eval = Evaluation(SVMMultiRegressorOutput , prp)
print (f"train RMSE score of {SVMMultiRegressorOutput['name']}: {eval.get_train_rmse_score()}")
print (f"test RMSE score of {SVMMultiRegressorOutput['name']}: {eval.get_test_rmse_score()}")
#%%
MLPRegressionSingleOutput = nnet_single_reg(prp, 3 , config, "MLPRegressionSingleOutput")
MLPRegressionSingleOutput['pred'] = np.argmax(MLPRegressionSingleOutput['pred'], axis=1) + 1 
MLPRegressionSingleOutput['fit'] = np.argmax(MLPRegressionSingleOutput['fit'], axis=1) + 1
eval = Evaluation(MLPRegressionSingleOutput , prp)
print (f"train RMSE score of {MLPRegressionSingleOutput['name']}: {eval.get_train_rmse_score()}")
print (f"test RMSE score of {MLPRegressionSingleOutput['name']}: {eval.get_test_rmse_score()}")
#%%
MLPRegressionMultiOutput = nnet_multi_reg(prp, config, np.random.randint(0, 5, len(prp.y_train)), "MLPRegressionMultiOutput")
MLPRegressionMultiOutput['pred'] = np.argmax(MLPRegressionMultiOutput['pred'], axis=1) + 1 
MLPRegressionMultiOutput['fit'] = np.argmax(MLPRegressionMultiOutput['fit'], axis=1) + 1
eval = Evaluation(MLPRegressionMultiOutput , prp)
print (f"train RMSE score of {MLPRegressionMultiOutput['name']}: {eval.get_train_rmse_score()}")
print (f"test RMSE score of {MLPRegressionMultiOutput['name']}: {eval.get_test_rmse_score()}")
