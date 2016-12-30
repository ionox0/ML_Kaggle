
# coding: utf-8

# In[5]:

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


train = pd.read_csv('data.csv')
quiz = pd.read_csv('quiz.csv')


# In[6]:

# Name Columns (53 total)
alphabet = list(string.ascii_lowercase)
alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']

train.columns = alphabet2
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
cols = train.columns
num_cols = train._get_numeric_data().columns
list(set(cols) - set(num_cols))

categorical_cols = ['a', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k',
 'l', 'm', 'n', 'o', 
   'ss', 'tt', 'uu'
 ]

for col in categorical_cols:
    train[col] = train[col].astype('category')
    quiz[col] = quiz[col].astype('category')

# Designate Numeric Columns (37 total)
numeric_cols = ['b', 'g', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
       'z', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii',
       'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'vv',
       'ww', 'xx', 'yy', 'zz']

continuous_cols = ['vv', 'ww']

numeric_indices = []
for i, letter in enumerate(alphabet2):
    if letter in numeric_cols:
        numeric_indices.append(i)
    
# [1, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
# 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52]

train_labels = np.array(train['aaa']).astype(int)


# In[3]:

##########################################
#          TheMapTaskClassifier          #
##########################################


class MetaClassifier:
    def __init__(self):    
        self.clf1 = KNeighborsClassifier()
        self.clf2 = LogisticRegression()
        self.clf3 = RandomForestClassifier(n_estimators=100, max_features=50, max_depth=200)
        self.clf4 = AdaBoostClassifier()
        self.clf5 = GaussianNB()
        

    def fit(self, train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_labels):
        self.clf1_trained = self.clf1.fit(train_data_1, train_labels)
        print('clf1 fitted')
#         self.clf2_trained = self.clf2.fit(train_data_2, train_labels)
#         print('clf2 fitted')
        self.clf3_trained = self.clf3.fit(train_data_3, train_labels)
        print('clf3 fitted')
        self.clf4_trained = self.clf4.fit(train_data_4, train_labels)
        print('clf4 fitted')
        self.clf5_trained = self.clf5.fit(train_data_5, train_labels)
        print('clf5 fitted')
        
        return self
        
    def predict(self, x_test_1, x_test_2, x_test_3, x_test_4, x_test_5):
        preds1 = self.clf1.predict(x_test_1)
        print('clf1 predicted')
#         preds2 = self.clf2.predict(x_test_2)
#         print('clf2 predicted')
        preds3 = self.clf3.predict(x_test_3)
        print('clf3 predicted')
        preds4 = self.clf4.predict(x_test_4)
        print('clf4 predicted')
        preds5 = self.clf5.predict(x_test_5)
        print('clf5 predicted')
    
        # Take sum and round pred results across all clf
        preds = np.sum(np.vstack([preds1,preds3,preds4,preds5]), axis=0)
        preds[preds >= 0] = 1
        preds[preds < 0] = -1
        
        return preds
    
    def get_params(self, deep=False):
        '''
        Hack to make scikit happy when using this class in scikitlearn.cross_val_score
        '''
        return {}


    
x_train, x_test, y_train, y_test = train_test_split(train, train.ix[:,-1], train_size=0.2, test_size=0.1)


train_dummies = pd.get_dummies(x_train[categorical_cols + zero_one_two_cols + boolean_cols])
quiz_dummies = pd.get_dummies(x_test[categorical_cols + zero_one_two_cols + boolean_cols])

train_dummies = train_dummies[[col for col in train_dummies.columns if col in quiz_dummies.columns]]
quiz_dummies = quiz_dummies[[col for col in quiz_dummies.columns if col in train_dummies.columns]]

print(np.array(train_dummies).shape)
a = MetaClassifier()


# In[47]:

print(x_train[continuous_cols].shape)
print(x_test[continuous_cols].shape)

a.fit(
    x_train[continuous_cols],
    x_train[continuous_cols],
    train_dummies,
    train_dummies,
    train_dummies,
    x_train.ix[:,-1]
)

train_preds = a.predict(
    x_test[continuous_cols],
    x_test[continuous_cols],
    quiz_dummies,
    quiz_dummies,
    quiz_dummies
)

print(train_preds.shape)
print(y_test.shape)


# In[49]:

err = (train_preds - y_test).sum() * 1.0 / len(train_preds)

print(err)

# KNN, GNB & RF
# on cont, cat_dum & cat_dum
# 0.1452% error


# In[ ]:

def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])


# In[ ]:

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


# In[ ]:

preds = clf_trained.predict(x_test)
print(confusion_matrix(y_test, preds, labels=[-1, 1]))

# Before class weighting
#  [27476   986]
#  [ 2401 19872]

# After class weighting
# [26350  2116]
# [ 1730 20539]


# In[ ]:

len(preds)
write_results(preds)


# In[7]:

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


# In[4]:

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



