
# coding: utf-8

# In[1]:

import csv
import math
import numpy as np
from numpy import genfromtxt
import pandas
import string
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.io import loadmat

# Classification utils
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_moons, make_circles, make_classification

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import *
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Load
train = pandas.read_csv('data.csv')
test = pandas.read_csv('quiz.csv')


# In[2]:

# Name Columns (53 total)
alphabet = list(string.ascii_lowercase)
alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']

train.columns = alphabet2
# Leave out label column for test data
test.columns = alphabet2[:-1]

# Designate Boolean Columns (15 total)
boolean_cols = [
    'g', 'p', 'q', 's',
    'v', 'w', 'y', 'z',
    'oo', 'pp', 'qq', 'rr',
    'xx', 'yy', 'zz'
]

# Designate Categorical Columns (16 total)
cols = train.columns
num_cols = train._get_numeric_data().columns
list(set(cols) - set(num_cols))

categorical_cols = ['a', 'c', 'e', 'd', 'f',
 'uu', 'i', 'k', 'j', 'm',
 'l', 'o', 'n', 'ss', 'h',
 'tt']

for col in categorical_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# Designate Numeric Columns (37 total)
numeric_cols = ['b', 'g', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
       'z', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii',
       'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'vv',
       'ww', 'xx', 'yy', 'zz', 'aaa']

numeric_indices = []
for i, letter in enumerate(alphabet2):
    if letter in numeric_cols:
        numeric_indices.append(i)
    
# [1, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
# 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52]


# In[3]:

##########################
### Data Preprocessing ###
##########################

# Normalizing data (just numeric columns)
train_std = StandardScaler().fit_transform(train[numeric_cols])
test_std = StandardScaler().fit_transform(test[numeric_cols[:-1]])

train_std = pandas.DataFrame(data=train_std[0:,0:])
test_std = pandas.DataFrame(data=test_std[0:,0:])

train_std.columns = numeric_cols
# Leave out label column for test data
test_std.columns = numeric_cols[:-1]


# In[ ]:

# SVD
u,s,v = np.linalg.svd(train_std.T)

print('SVD: ', u)


# In[ ]:

# Eigendecomposition
cov_mat = np.cov(train_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[ ]:

# Explained var
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Graph explained variance of eigenvectors
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    
    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid', label='cumulative explained variance')
    
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
# Results:
# Determining relevance of eigenvalues of the covariance matrix for just numerical columns
# 1st 5 - 42% of variance
# 1st 17 - 80%
# 1st 20 - 90%  --> use 20.


# In[ ]:

# PCA Shortcut
pca = PCA(n_components=2)
Y_sklearn = pca.fit_transform(train_std)
print(Y_sklearn)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):
        plt.scatter(Y_sklearn['aaa'==lab, 0], Y_sklearn['aaa'==lab, 1], label=lab, c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()


# In[15]:

#########################
### Feature Selection ###
#########################

# This sections calculates:
#   reduced_features_train
#   rfe_selected_numeric_cols

# Remove features with low variance
sel = VarianceThreshold(threshold=(.8))
reduced_features_train = sel.fit_transform(train_std)
print(reduced_features_train)
# print(reduced_features_train)

# Use Recursive Feature Selection
# Must choose estimator here - todo, try alternatives
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=7, step=1)
rfe.fit(train[numeric_cols], train['aaa'])
# ranking = rfe.ranking_.reshape(train[numeric_cols].shape)

# for i in range(37):
#     print(i, rfe.ranking_[i])

# Results
# (0, 27) (1, 16) (2, 14) (3, 4) (4, 12) (5, 17) (6, 9) (7, 6) (8, 5) (9, 20)
# (10, 2) (11, 1) (12, 3) (13, 1) (14, 1) (15, 1) (16, 15) (17, 11)
# (18, 24) (19, 19) (20, 26) (21, 31) (22, 29) (23, 1) (24, 13) (25, 18) (26, 8)
# (27, 10) (28, 21) (29, 1) (30, 7) (31, 28) (32, 30) (33, 23) (34, 22)
# (35, 25) (36, 1)

# Best numeric cols (rank 1):
# 11, 13, 14, 15, 23, 29, 36
rfe_selected_numeric_cols = ['l',  'n',  'x',  'p',  'x',  'dd', 'kk']


# In[16]:

# Check out coorelation matrix of vars
train.corr()

# Notable correlations with 'aaa' label:
# q 9%
# aa 20%
# bb 17%
# vv 41%
# ww 41%

coorelated_features = ['q', 'aa', 'bb', 'vv', 'ww']


# In[ ]:

# Check out covariance matrix of vars
cov = np.cov(train_std.T)
cov_df = pandas.DataFrame(data=cov)
# print(df)

# Print in sorted order
s = cov_df.unstack()
so = s.order(kind="quicksort")
print(so)

# Notable findings:
# cols 9, 5 -> 100% coorelation
# cols 2 + 3, 29 -> 62%, 48%
# cols 1 + 2, 9 -> -76%, -60% 
# 27, 28 - 41%
# 2 + 3, 28 + 29
# 31, 32 - 99%
# 33, 34, - 89%
# 31 + 32, 36 - 41%


# In[ ]:

# Give higher weights to more coorelated variables - todo


# In[14]:

################################
### Train + Test Classifiers ###
################################

# Options:
# use train_std (standardized columns)
# use reduced_features_train (removed features with little variance)
# use train[rfe_selected_numeric_cols] (RFE reduced features)

def cross_val(clf, train_data, train_labels):
    error_rates = []
    for i in range(3):
        x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                               train_labels,
                                                               test_size=0.4)
        clf_trained = clf().fit(x_train, y_train)
        e = clf_trained.score(x_test, y_test)
        error_rates.append(e)
    
    return np.mean(error_rates)


def meta_classify(train, test, train_std, test_std):
    '''
    Make predictions from `test` features using the classifier that has the lowest error on the `training` features.
    Datasets should get correct headers from previous code.
    This method depends on the previous `cross_val` method.
    '''
    
    numerical_clf = [
        Lasso,
#         DecisionTreeClassifier,
#         RandomForestClassifier,
        LogisticRegression, 
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
        AdaBoostClassifier,
        KNeighborsClassifier,  # takes a while...
#         SVC,    # takes forever.....
    ]
    
    categorical_clf = [
#         GaussianNB,
        MultinomialNB,
    ]

    err_rates = []
    for clf in numerical_clf:
        # Keep label out of features
        labels = np.array(train_std['aaa']).astype(int)
        e = cross_val(clf, train_std[numeric_cols[:-1]], labels)
        err_rates.append(e)
        print(clf, e)
              
#     for clf in categorical_clf:
#         e = cross_val(clf, train[categorical_cols], train.ix[:,36])
#         err_rates.append(e)
#         print(clf, e)
            
    best_clf = numerical_clf[np.argmin(err_rates)]
    print('Best: ', best_clf)
    trained = best_clf().fit(train_std[numeric_cols[:-1]], train_std['aaa'])
    return trained.predict(np.array(test[numeric_cols[:-1]]))
    
    
# print(train.shape)
# print(reduced_features_train.shape)
# print(train[rfe_selected_numeric_cols])

# train, test - pandas
# train_std, test_std - pandas


preds = meta_classify(train, test, train_std, test_std)
preds[preds > 0] = 1
preds[preds < 0] = -1
preds = preds.astype(int)
write_results(preds)


# In[ ]:

##########################
### Tune params - todo ###
##########################

DecisionTreeClassifier(max_depth=5),
RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
SVC(kernel="linear", C=0.025)


# In[ ]:

# Method to print predicted test labels formatted for kaggle submission
def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])


# In[ ]:




# In[ ]:

# Misc. functions

cross_val(KNeighborsClassifier, np.array(train[numeric_cols[:-1]]), np.array(train['aaa']))
# --> 11%

knn_clf = KNeighborsClassifier()
knn_clf.fit(train[numeric_cols[:-1]], train['aaa'])
preds = knn_clf.predict(np.array(test[numeric_cols[:-1]]))
write_results(preds)

