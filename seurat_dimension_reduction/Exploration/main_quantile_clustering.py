#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import glob
import math
import time
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import scipy as sc
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import random
import seaborn as sns; sns.set_theme()
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import statsmodels.api as sm
import pylab as pl
from matplotlib.pyplot import figure
from IPython import display
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import umap
from sklearn import svm
from lightgbm import LGBMClassifier
from numpy import std
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained

train = pd.read_csv('./MLR_Project_train.csv')
test = pd.read_csv('./MLR_Project_test.csv')
# In[11]:
train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
test = test.loc[:, ~test.columns.str.contains('^Unnamed')]
# In[24]:
train_test = pd.concat([train, test], axis=0)
# In[26]:
train_test_np = train_test.iloc[:, :66].to_numpy()
# In[22]:
kmeans = KMeans(n_clusters=100, random_state=0).fit(train_test_np)
train_np = train_test.iloc[:90000, :66].to_numpy()
test_np = train_test.iloc[90000:, :66].to_numpy()

train_pred = kmeans.predict(train_np)
test_pred = kmeans.predict(test_np)
train['pred_class'] = train_pred
test['pred_class'] = test_pred
# In[27]:

train_test_np_cons = train_test_np

clf = KMeansConstrained(
     n_clusters=1000,
     size_min=8,
     size_max=125,
     random_state=0
)
clf.fit_predict(train_test_np_cons)
print(clf.labels_)

train_test['pred_class'] = clf.labels_

quantiles_num = []
train_sum_returns = []
test_sum_returns = []

models = []

for i in range(1000):
    
    train_seg = train[train['pred_class']==i]
    test_seg = test[test['pred_class']==i]
    quantiles_num.append([train_seg.shape[0], test_seg.shape[0]])
    
    if train_seg.shape[0] == 1:
        continue
    if test_seg.shape[0] < 1:
        continue
    reg = Ridge(alpha=0.5).fit(pd.DataFrame(train_seg.iloc[:, :66]), train_seg['TARGET'])
    
    models.append(reg)
    
    train_pred = reg.predict(pd.DataFrame(train_seg.iloc[:, :66]))

    test_pred = reg.predict(pd.DataFrame(test_seg.iloc[:, :66]))

    train_res = np.sum(train_seg['TARGET'][train_pred>0])
    test_res = np.sum(test_seg['TARGET'][test_pred>0])
    
    train_sum_returns.append(train_res)
    test_sum_returns.append(test_res)

print(sum(train_sum_returns))
print(sum(test_sum_returns))