
# coding: utf-8

# In[87]:

import csv
import math
import numpy as np
import pandas as pd
import string
# Classification utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Classifiers
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import VotingClassifier

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
import pprint
pp = pprint.PrettyPrinter(indent=4)

train = pd.read_csv('data.csv')
quiz = pd.read_csv('quiz.csv')


# In[100]:

# Name Columns (53 total)
alphabet = list(string.ascii_lowercase)
alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']

train.columns = alphabet2
quiz.columns = alphabet2[:-1]

continuous_cols = ['vv', 'ww']
boolean_cols = ['g', 'p', 'q', 's', 'v', 'w', 'y', 'z', 'oo', 'pp', 'qq', 'rr', 'xx', 'yy', 'zz']
zero_one_two_cols = ['aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk','ll','mm','nn']
categorical_cols = ['a', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ss', 'tt', 'uu']
numeric_cols = ['b', 'g', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
       'z', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii',
       'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'vv',
       'ww', 'xx', 'yy', 'zz']

cols = train.columns
num_cols = train._get_numeric_data().columns
list(set(cols) - set(num_cols))

for col in categorical_cols:
    task[col] = train[col].astype('category')
    quiz[col] = quiz[col].astype('category')

numeric_indices = []
for i, letter in enumerate(alphabet2):
    if letter in numeric_cols:
        numeric_indices.append(i)

train_labels = np.array(task['aaa']).astype(int)


# In[169]:

# Test prediction
x_train, x_test, y_train, y_test = train_test_split(train, train.ix[:,-1], train_size=0.1, test_size=0.1)

pf = PolynomialFeatures(degree=2)
train_cont_expanded = pf.fit_transform(x_train[continuous_cols])
test_cont_expanded = pf.fit_transform(x_test[continuous_cols])

train_cont_expanded = pd.DataFrame(data=train_cont_expanded)
test_cont_expanded = pd.DataFrame(data=test_cont_expanded)

train_dummies = pd.get_dummies(x_train[categorical_cols + zero_one_two_cols + boolean_cols])
test_dummies = pd.get_dummies(x_test[categorical_cols + zero_one_two_cols + boolean_cols])

train_dummies = train_dummies[[col for col in train_dummies.columns if col in test_dummies.columns]]
test_dummies = test_dummies[[col for col in test_dummies.columns if col in train_dummies.columns]]

train_dummies_plus_continuous = pd.concat([train_dummies, train_cont_expanded], axis=1, ignore_index=True)
test_dummies_plus_continuous = pd.concat([test_dummies, test_cont_expanded], axis=1, ignore_index=True)

print(train_dummies.shape)
print(train_cont_expanded.shape)


# In[171]:

rf = RandomForestClassifier(n_estimators=200, n_jobs=3)
# rf = ExtraTreesRegressor(n_estimators=10, max_features=50, max_depth=200)

print(train_dummies_plus_continuous)
print(test_dummies_plus_continuous)

print(train_dummies_plus_continuous.shape)
print(y_train.shape)

rf.fit(train_dummies_plus_continuous, y_train)

scores = cross_val_score(rf, test_dummies_plus_continuous, y_test, cv=5)
print(scores)



# rf = RandomForestClassifier(n_estimators=200, max_features=50, max_depth=200)
# [ 0.90307329  0.89791092  0.90776508  0.90930599  0.91285489]

# rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, max_features=50, max_depth=200)
# [ 0.9168309   0.91486007  0.90500591  0.90579425  0.91916404]

# rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features=50, max_depth=250)
# [ 0.89948758  0.90145842  0.89515175  0.90303508  0.90891167]

# rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features=500, max_depth=500)
# [ 0.90185258  0.90776508  0.89948758  0.90421758  0.91009464]

# rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features=500, n_jobs=3)
# [ 0.89598109  0.89869925  0.9144659   0.9022082   0.90812303]

# rf = RandomForestClassifier(n_estimators=200, max_features=500, n_jobs=3)
# [ 0.91643674  0.91170674  0.90540008  0.90973591  0.90891167]

# rf = RandomForestClassifier(n_estimators=200, n_jobs=3)
# [ 0.9251084   0.90934174  0.91131257  0.91328341  0.91088328]

# rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features=500, max_depth=500)
# preds = rf.predict(quiz_dummies_plus_continuous)


# In[115]:

# Actual prediction
train_dummies = pd.get_dummies(train[categorical_cols + zero_one_two_cols + boolean_cols])
quiz_dummies = pd.get_dummies(quiz[categorical_cols + zero_one_two_cols + boolean_cols])

train_dummies = train_dummies[[col for col in train_dummies.columns if col in quiz_dummies.columns]]
quiz_dummies = quiz_dummies[[col for col in quiz_dummies.columns if col in train_dummies.columns]]

train_dummies_plus_continuous = pd.concat([train_dummies, train[continuous_cols]], axis=1)
quiz_dummies_plus_continuous = pd.concat([quiz_dummies, quiz[continuous_cols]], axis=1)

rf = RandomForestClassifier(n_estimators=100, max_features=50, max_depth=200)
rf.fit(train_dummies_plus_continuous, train.ix[:,-1])
preds = rf.predict(quiz_dummies_plus_continuous)

print(quiz.shape)
print(quiz_dummies.shape)
print(preds.shape)


# In[61]:

def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])
            
write_results(preds)


# In[ ]:



