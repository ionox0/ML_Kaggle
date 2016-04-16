
# coding: utf-8

# In[42]:

import csv
import math
import numpy as np
import pandas as pd
import string

# Classification utils
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import VotingClassifier

# Set ipython's max row / column display
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 50)


task = pandas.read_csv('data.csv')
quiz = pandas.read_csv('quiz.csv')


# In[43]:

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
        
        self.clf1 = KNeighborsClassifier()
        self.clf2 = RandomForestClassifier()
        self.clf3 = GaussianNB()
        

    def fit(self, train_data, train_labels):
        enc_train = encode_as_labels(train_data[categorical_cols])
        self.clf1_trained = self.clf1.fit(train_data[self.numeric_cols], train_labels)
        self.clf2_trained = self.clf2.fit(train_data[self.numeric_cols], train_labels)
        self.clf3_trained = self.clf3.fit(enc_train, train_labels)
        return self
        
    def predict(self, data):
        enc_test = pandas.get_dummies(data[categorical_cols])
        
        preds1 = self.clf1.predict(data[numeric_cols])
        preds2 = self.clf2.predict(data[numeric_cols])
        preds3 = self.clf3.predict(enc_test)

        preds = np.sum(np.vstack([preds1,preds2,preds3]), axis=0)
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
write_results(preds)


# In[ ]:

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


# In[30]:

def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])


# In[ ]:

df = pandas.get_dummies(train[categorical_cols])
c = RandomForestClassifier(max_depth=200, max_features=1, n_estimators=500)
# bool_num = np.concatenate([boolean_cols, numeric_cols])
x_train, x_test, y_train, y_test = train_test_split(df, train.ix[:,-1], test_size=0.4)
clf_trained = c.fit(x_train, y_train)
cross_val_score(clf_trained, x_test, y_test, cv=5) # array([ 0.91741564,  0.91528364]) - v. good


# In[ ]:

c = RandomForestClassifier(max_depth=200, max_features=1, n_estimators=500)

train_df = pandas.get_dummies(train[categorical_cols])
clf_trained = c.fit(train_df, train.ix[:,-1])

# Figure out how to add empty columns for 1783 cols in train_df that aren't in test_df
test_df = pandas.get_dummies(test[categorical_cols])
preds = clf_trained.predict(test_df)


# In[ ]:




# In[44]:

# Peter's method

X_dummies = pd.get_dummies(task[categorical_cols + zero_one_two_cols + boolean_cols])
X_quiz_dummies = pd.get_dummies(quiz[categorical_cols + zero_one_two_cols + boolean_cols])

X_train_dummies = X_dummies[[col for col in X_dummies.columns if col in X_quiz_dummies.columns]]
X_quiz_dummies = X_quiz_dummies[[col for col in X_quiz_dummies.columns if col in X_train_dummies.columns]]

# Added class weights b/c we're overpredicting on the -1's
clf = RandomForestClassifier(max_depth=200, max_features=1, n_estimators=500, class_weight={-1: 1, 1: 2}, n_jobs=-1)

x_train, x_test, y_train, y_test = train_test_split(X_train_dummies, task.ix[:,-1], test_size=0.4)
clf_trained = clf.fit(x_train, y_train)
scores = cross_val_score(clf_trained, x_test, y_test, cv=5)
print(scores)
# [0.92550256 0.92313756 0.927959 0.92667061 0.92312241] (before class weighting)
# [0.92047694 0.92056766 0.91682271 0.91652705 0.91257638] (after class weighting) (worse)
return

clf_full_trained = clf.fit(X_train_dummies, task.ix[:,-1])
preds = clf_full_trained.predict(X_quiz_dummies)


# In[45]:

preds = clf_trained.predict(x_test)
print(confusion_matrix(y_test, preds, labels=[-1, 1]))

# Before class weighting
#  [27476   986]
#  [ 2401 19872]

# After class weighting
# [26350  2116]
# [ 1730 20539]


# In[31]:

len(preds)
write_results(preds)


# In[ ]:

###########################
### Tweaking and tuning ###
###########################

# Normalizing data (just numeric columns)
train_std = StandardScaler().fit_transform(train[numeric_cols])
test_std = StandardScaler().fit_transform(test[numeric_cols[:-1]])

train_std = pandas.DataFrame(data=train_std[0:,0:])
test_std = pandas.DataFrame(data=test_std[0:,0:])

train_std.columns = numeric_cols
# Leave out label column for test data
test_std.columns = numeric_cols[:-1]


# In[ ]:

sel = VarianceThreshold(threshold=(.8))
reduced_features_train = sel.fit_transform(train_std)

rfe = RFE(estimator=LogisticRegression(), n_features_to_select=7, step=1)
rfe.fit(train[numeric_cols], train['aaa'])


# In[ ]:

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

