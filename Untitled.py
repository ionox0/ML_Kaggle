
# coding: utf-8

# In[24]:

import csv
import math
import numpy as np
import pandas as pd
import string
# Classification utils
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
import pprint
pp = pprint.PrettyPrinter(indent=4)

task = pd.read_csv('data.csv')
quiz = pd.read_csv('quiz.csv')


# In[26]:

# Name Columns (53 total)
alphabet = list(string.ascii_lowercase)
alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']

task.columns = alphabet2
quiz.columns = alphabet2[:-1]

continuous_cols = ['vv', 'ww']
boolean_cols = ['g', 'p', 'q', 's', 'v', 'w', 'y', 'z', 'oo', 'pp', 'qq', 'rr', 'xx', 'yy', 'zz']
zero_one_two_cols = ['aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk','ll','mm','nn']
categorical_cols = ['a', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ss', 'tt', 'uu']
numeric_cols = ['b', 'g', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
       'z', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii',
       'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'vv',
       'ww', 'xx', 'yy', 'zz']

cols = task.columns
num_cols = task._get_numeric_data().columns
list(set(cols) - set(num_cols))

for col in categorical_cols:
    task[col] = task[col].astype('category')
    quiz[col] = quiz[col].astype('category')

numeric_indices = []
for i, letter in enumerate(alphabet2):
    if letter in numeric_cols:
        numeric_indices.append(i)

train_labels = np.array(task['aaa']).astype(int)


# In[27]:

train_dummies = pd.get_dummies(task[categorical_cols + zero_one_two_cols + boolean_cols])
quiz_dummies = pd.get_dummies(quiz[categorical_cols + zero_one_two_cols + boolean_cols])

train_dummies = train_dummies[[col for col in train_dummies.columns if col in quiz_dummies.columns]]
quiz_dummies = quiz_dummies[[col for col in quiz_dummies.columns if col in train_dummies.columns]]

train_dummies_plus_continuous = pd.concat([train_dummies, task[continuous_cols]], axis=1)


# In[29]:

x_train, x_test, y_train, y_test = train_test_split(task, task.ix[:,-1], train_size=0.8, test_size=0.05)

rf = RandomForestClassifier(n_jobs=3, n_estimators=100, max_features=50, max_depth=200)
x_train, x_test, y_train, y_test = train_test_split(train_dummies_plus_continuous, task.ix[:,-1], train_size=0.8, test_size=0.05)
rf_trained = rf.fit(x_train, y_train)

preds = rf.predict()


log = LogisticRegression()
log_trained = log.fit(x_train[continuous_cols], y_train)

cross_val_score(log_trained, x_test, y_test, cv=2)

preds_2 = log.predict(x_test)


# In[21]:

task


# In[ ]:



