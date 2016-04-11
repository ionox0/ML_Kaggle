
# coding: utf-8

# In[23]:

import csv
import math
import numpy as np
import pandas
import string

# Classification utils
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.ensemble import VotingClassifier


# In[2]:

# Load
train = pandas.read_csv('data.csv')
test = pandas.read_csv('quiz.csv')


# In[20]:

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

train_labels = np.array(train['aaa']).astype(int)


# In[ ]:

sel = VarianceThreshold(threshold=(.8))
reduced_features_train = sel.fit_transform(train[numeric_cols[:-1]])
print(reduced_features_train)


# In[14]:

# Features shown to have coorelation with label
coorelated_features = ['q', 'aa', 'bb', 'vv', 'ww']
# Features selected by Recursive Feature Extraction
rfe_selected_numeric_cols = ['x', 'p', 'dd', 'kk']


# In[7]:

def cross_val(clf, train_data, train_labels):
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.4)
    clf_trained = clf.fit(x_train, y_train)
    scores = cross_val_score(clf_trained, x_train, y_train, cv=2)
    return scores


# In[18]:

# Training error using numeric columns

# Keep label out of features
clf = KNeighborsClassifier()
e = cross_val(clf, train[coorelated_features], train_labels)
print(e)


# In[12]:

# Training error using categorical columns

# Method to convert to one-hot encodings
# pd.get_dummies --> returns matrix of every feature, concatenated from all cols, into one feature space
def encode_as_labels(X):
    output = X.copy()
    if X.columns is not None:
        for col in X.columns:
            output[col] = LabelEncoder().fit_transform(output[col])
    else:
        for colname,col in output.iteritems():
            output[colname] = LabelEncoder().fit_transform(col)
    return output

# kf = KFold(4, n_folds=2, shuffle=True)

enc_train = encode_as_labels(train[categorical_cols])

r = RandomForestClassifier(n_estimators=100, max_depth=5)

e = cross_val(r, enc_train, train_labels)
print(e)


# In[21]:

# Combine numerical and categorical classifiers
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.4)

num_clf = KNeighborsClassifier()
num_clf_trained = num_clf.fit(train[coorelated_features], train_labels)

enc_train = encode_as_labels(train[categorical_cols])
cat_clf = RandomForestClassifier(n_estimators=100, max_depth=5)
cat_clf_trained = cat_clf.fit(enc_train, train_labels)

scores = cross_val_score(num_clf_trained, x_train, y_train, cv=2)


# In[17]:

# Make predictions using numeric columns
trained = KNeighborsClassifier().fit(train[numeric_cols[:-1]], train['aaa'])
preds = trained.predict(np.array(test[numeric_cols[:-1]]))
write_results(preds)


# In[ ]:

# Make predictions using categorical columns
enc_train = encode_as_labels(train[categorical_cols])
frames = [train[numeric_cols], enc_train]
result = pandas.concat(frames, axis=1)

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1, max_depth=5)
clf3 = GaussianNB()

eclf = VotingClassifier([('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
scores = cross_val_score(eclf, result, labels, cv=10, scoring='accuracy')
scores


# In[10]:

def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])


# In[36]:

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1, max_depth=5)
clf3 = GaussianNB()

eclf = VotingClassifier([('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1,1,1])

enc_train = encode_as_labels(train[categorical_cols])
frames = [train[numeric_cols[:-1]], enc_train]
train_result = pandas.concat(frames, axis=1)

enc_test = encode_as_labels(test[categorical_cols])
frames = [test[numeric_cols[:-1]], enc_test]
test_result = pandas.concat(frames, axis=1)

eclf.fit(train_result, train['aaa'])
write_results(eclf.predict(test_result))


# In[ ]:



