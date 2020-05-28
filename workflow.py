"""This script details the workflow of getting the results for the
sklearn models and importing all the functions needed from all .py files"""

"""Some of the models does implement combinatorial grid search cv hyperparamter tuning
to pick the best model Paramters and check against overfitting, so it takes a while to run"""



%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clean import cleaned
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from transformation import preprocess,normalisation
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import itertools
from random_forest import smote_random
from xgb_model import xgb_smote_recall
from confusion_mx import confusion
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc,precision_recall_curve
from Logisticresults import baseline_regression,smote_logistic


transactions = pd.read_csv('transactions_obf.csv')
labels = pd.read_csv('labels_obf.csv')

cleaned(transactions,labels)

cat_features,cont_features,target = preprocess(transactions,labels)


full_df = normalisation(cat_features,cont_features)


baseline_regression(full_df,target)


smote_logistic(full_df,target)



smote_random(full_df,target)


xgb_smote_recall(full_df,target)

"""After training four different models on the train data, validating with k-folds and testing
on test data, the best performing model is the xgb classifier test set recall - 80.84%"""
