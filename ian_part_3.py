
# coding: utf-8

# In[2]:

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


# In[3]:

# Load
train = pandas.read_csv('data.csv')
test = pandas.read_csv('quiz.csv')


# In[10]:

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
       'ww', 'xx', 'yy', 'zz']

numeric_indices = []
for i, letter in enumerate(alphabet2):
    if letter in numeric_cols:
        numeric_indices.append(i)
    
# [1, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
# 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52]

train_labels = np.array(train['aaa']).astype(int)


# In[11]:

##########################################
#          TheMapTaskClassifier          #
##########################################

# VotingClassifier doesn't let us use different classifiers on different columns,
# they all have to work on all cols...

class MetaClassifier:
    def __init__(self):
        self.numeric_cols = ['b', 'g', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
           'z', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii',
           'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'vv',
           'ww', 'xx', 'yy', 'zz']
        
        self.categorical_cols = ['a', 'c', 'e', 'd', 'f',
            'uu', 'i', 'k', 'j', 'm', 'l', 'o', 'n', 'ss', 'h', 'tt']
        
        self.actual_numeric_cols = ['vv', 'ww']
        
        self.boolean_cols = [
            'g', 'p', 'q', 's',
            'v', 'w', 'y', 'z',
            'oo', 'pp', 'qq', 'rr',
            'xx', 'yy', 'zz']
        
        self.clf1 = LogisticRegression(random_state=1)
        self.clf2 = RandomForestClassifier(random_state=1, max_depth=5)
        self.clf3 = GaussianNB()
        

    def fit(self, train_data, train_labels):
        enc_train = encode_as_labels(train_data[categorical_cols])
        self.clf1_trained = self.clf1.fit(train_data[self.numeric_cols], train_labels)
        self.clf2_trained = self.clf2.fit(train_data[self.numeric_cols], train_labels)
        self.clf3_trained = self.clf3.fit(enc_train, train_labels)
        return self
        
    def predict(self, data):
        enc_test = encode_as_labels(data[categorical_cols])
        
        preds1 = self.clf1.predict(data[numeric_cols])
        preds2 = self.clf2.predict(data[numeric_cols])
        preds3 = self.clf3.predict(enc_test)

        preds = np.sum(np.vstack([preds1,preds2,preds3]), axis=0)
        print('before rounding: ', preds)
        preds[preds > 0] = 1
        preds[preds < 0] = -1
        
        return preds
    
    def get_params(self, deep=False):
        '''
        Hack to make scikit happy when using this class in scikitlearn.cross_val_score
        '''
        return {}
    

def cross_val(clf, train_data, train_labels):
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.4)
    clf_trained = clf.fit(x_train, y_train)
    scores = cross_val_score(clf_trained, x_train, y_train, cv=4, scoring="accuracy")
    return scores


a = MetaClassifier()
preds = cross_val(a, train.ix[:,:-1], train['aaa'])
print(preds)
write_results(preds)


# In[6]:

# Method to convert to one-hot encodings for categorical variables
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


# In[7]:

def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])


# In[ ]:



