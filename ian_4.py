
# coding: utf-8

# In[4]:

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
from sklearn import grid_search

# Classifiers
from sklearn.ensemble import RandomForestClassifier

task = pd.read_csv('data.csv')
quiz = pd.read_csv('quiz.csv')


# In[6]:

# Name Columns (53 total)
alphabet = list(string.ascii_lowercase)
alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']

task.columns = alphabet2
# Leave out label column for test data
quiz.columns = alphabet2[:-1]

# Designate Boolean Columns (15 total)
boolean_cols = [
    'g', 'p', 'q', 's',
    'v', 'w', 'y', 'z',
    'oo', 'pp', 'qq', 'rr',
    'xx', 'yy', 'zz'
]

zero_one_two_cols = ['aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk','ll','mm','nn']

# Designate Categorical Columns (16 total)
cols = task.columns
num_cols = task._get_numeric_data().columns
list(set(cols) - set(num_cols))

categorical_cols = ['a', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k',
 'l', 'm', 'n', 'o', 
   'ss', 'tt', 'uu'
 ]

for col in categorical_cols:
    task[col] = task[col].astype('category')
    quiz[col] = quiz[col].astype('category')

# Designate Numeric Columns (37 total)
numeric_cols = ['b', 'g', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
       'z', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii',
       'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'vv',
       'ww', 'xx', 'yy', 'zz']

numeric_indices = []
for i, letter in enumerate(alphabet2):
    if letter in numeric_cols:
        numeric_indices.append(i)
    
# [1, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
# 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52]

train_labels = np.array(task['aaa']).astype(int)


# In[ ]:

# One-hot encoded features for categorical vars

X_dummies = pd.get_dummies(task[categorical_cols + zero_one_two_cols + boolean_cols])
X_quiz_dummies = pd.get_dummies(quiz[categorical_cols + zero_one_two_cols + boolean_cols])

X_train_dummies = X_dummies[[col for col in X_dummies.columns if col in X_quiz_dummies.columns]]
X_quiz_dummies = X_quiz_dummies[[col for col in X_quiz_dummies.columns if col in X_train_dummies.columns]]


# In[ ]:

# Exploring different parameter settings with grid_search
# Features reduced with select k best
# Training size reduced with train_test_split

X_train_k_best = SelectKBest(chi2, k=10).fit_transform(X_train_dummies, task.ix[:,-1])

param_grid = [
    {'max_depth': [1, 10, 100, 1000], 'max_features': [1, 5, 10], 'n_estimators': [1, 10, 100, 1000]}
]


rf = RandomForestClassifier(n_jobs=-1)
clf = grid_search.GridSearchCV(rf, param_grid)

x_train, x_test, y_train, y_test = train_test_split(X_train_k_best, task.ix[:,-1], train_size=0.1, test_size=0.3)
clf_trained = clf.fit(x_train, y_train)

scores = cross_val_score(clf_trained, x_test, y_test, cv=2)

print(scores)
print('best params: ', clf_trained.best_params_)


# In[ ]:



