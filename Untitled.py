
# coding: utf-8

# In[56]:

import csv
import math
import numpy as np
import pandas
import string

# Classification utils
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



# In[2]:

# Load
train = pandas.read_csv('data.csv')
test = pandas.read_csv('quiz.csv')


# In[3]:

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


# In[34]:

sel = VarianceThreshold(threshold=(.8))
reduced_features_train = sel.fit_transform(train[numeric_cols[:-1]])
print(reduced_features_train)


# In[44]:

# Features shown to have coorelation with label
coorelated_features = ['q', 'aa', 'bb', 'vv', 'ww']
# Features selected by Recursive Feature Extraction
rfe_selected_numeric_cols = ['x', 'p', 'dd', 'kk']


# In[4]:

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


# In[54]:

# Training error using numeric columns
labels = np.array(train['aaa']).astype(int)
# Keep label out of features
e = cross_val(DecisionTreeClassifier, train[coorelated_features], labels)
print(e)


# In[57]:

# Training error using categorical columns
labels = np.array(train['aaa']).astype(int)
e = cross_val(GaussianNB, train[categorical_cols], labels)
err_rates.append(e)
print(clf, e)


# In[53]:

# Make predictions using numeric columns
trained = KNeighborsClassifier().fit(train[numeric_cols[:-1]], train['aaa'])
preds = trained.predict(np.array(test[numeric_cols[:-1]]))
write_results(preds)


# In[13]:

def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])


# In[ ]:



