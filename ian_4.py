
# coding: utf-8

# In[1]:

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
from sklearn.metrics import f1_score

# Classifiers
from sklearn.ensemble import RandomForestClassifier

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
import pprint
pp = pprint.PrettyPrinter(indent=4)

task = pd.read_csv('data.csv')
quiz = pd.read_csv('quiz.csv')


# In[2]:

# Name Columns (53 total)
alphabet = list(string.ascii_lowercase)
alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']

task.columns = alphabet2
# Leave out label column for test data
quiz.columns = alphabet2[:-1]

continuous_cols = [
   'vv', 'ww'
]

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

train_labels = np.array(task['aaa']).astype(int)


# In[3]:

# One-hot encoded features for categorical vars

X_dummies = pd.get_dummies(task[categorical_cols + zero_one_two_cols + boolean_cols])
X_quiz_dummies = pd.get_dummies(quiz[categorical_cols + zero_one_two_cols + boolean_cols])

X_train_dummies = X_dummies[[col for col in X_dummies.columns if col in X_quiz_dummies.columns]]
X_quiz_dummies = X_quiz_dummies[[col for col in X_quiz_dummies.columns if col in X_train_dummies.columns]]


# In[5]:

# Select K best
k_best = SelectKBest(chi2, k=1000)
X_train_k_best_cols = k_best.fit_transform(X_train_dummies, task.ix[:,-1])
a = X_train_k_best_cols.get_support()

# Add the continuous features back in
X_train_k_best_cols = pd.DataFrame(X_train_k_best_cols)
X_train_k_best_cols = pd.concat([X_train_k_best_cols, task[continuous_cols]], axis=1)


# In[23]:

X_quiz_k_best_cols = X_quiz_dummies.iloc[:,a]

X_quiz_k_best = pd.DataFrame(X_quiz_k_best_cols)
X_quiz_k_best = pd.concat([X_quiz_k_best, quiz[continuous_cols]], axis=1)


# In[24]:

rf = RandomForestClassifier(n_jobs=3, n_estimators=100, max_features=50, max_depth=200)
clf_full_trained = rf.fit(X_train_k_best_cols, task.ix[:,-1])


# In[22]:

print(X_quiz_k_best)


# In[21]:

preds = clf_full_trained.predict(X_quiz_k_best)
write_results(preds)


# In[4]:

# Exploring different parameter settings with grid_search
# Features reduced with select k best
# Training size reduced with train_test_split

param_grid = [{
    'n_estimators': [100],
    'max_features': [50],
    'max_depth': [200]
}]

rf = RandomForestClassifier(n_jobs=2)
clf = grid_search.GridSearchCV(rf, param_grid)

x_train, x_test, y_train, y_test = train_test_split(X_train_k_best, task.ix[:,-1], train_size=0.05, test_size=0.05)
clf_trained = clf.fit(x_train, y_train)

scores = cross_val_score(clf_trained, x_test, y_test, cv=2)

print(scores)
print('best params: ', clf_trained.best_params_)


# In[ ]:

# n_estimators accuracy plot
param_results = clf_trained.grid_scores_

# Features were reduced using select K best (1000)
# train_size=0.05, test_size=0.05 (train_test_split)
n_estimators_values = [1, 10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
n_estimators_results = [0.65084, 0.81438, 0.85980, 0.86027, 0.86217, 0.86169, 0.86106, 0.86343,
                        0.86154, 0.86138, 0.86264, 0.86359, 0.86185]

ts = pd.Series(n_estimators_results, index=n_estimators_values)

ax = ts.plot()
ax.set_title('Number of RF estimators vs RF prediction accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('n_estimators')
ax.set_ylabel('accuracy')

plt.figure(); ts.plot();
plt.show()


# In[ ]:

# max_features accuracy plot
param_results = clf_trained.grid_scores_
# pp.pprint(param_results)

max_features_values = [1, 10, 50, 100, 200, 500, 1000]
max_features_results = [0.57562, 0.84608, 0.87352, 0.87053, 0.87478, 0.87305, 0.86942]

ts = pd.Series(max_features_results, index=max_features_values)

ax = ts.plot()
ax.set_title('Number of RF features vs RF prediction accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('max_features')
ax.set_ylabel('accuracy')

plt.figure(); ts.plot();
plt.show()


# In[ ]:

# max_depth accuracy plot
param_results = clf_trained.grid_scores_
pp.pprint(param_results)

max_depth_values = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000]
max_depth_results = [0.64517, 0.86501, 0.88850, 0.88771, 0.89182, 0.88992, 0.88945, 0.88693, 0.88992]

ts = pd.Series(max_depth_results, index=max_depth_values)

ax = ts.plot()
ax.set_title('RF max depth vs RF prediction accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('max_depth')
ax.set_ylabel('accuracy')

plt.figure(); ts.plot();
plt.show()


# In[1]:

def write_results(preds):
    with open('test_predictions.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'Prediction'])
        for i, pred in enumerate(preds):
            writer.writerow([i+1, pred])


# In[ ]:



