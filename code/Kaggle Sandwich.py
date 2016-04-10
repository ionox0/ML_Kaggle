
# coding: utf-8

# In[2]:

import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.linear_model import Lasso
import string
from sklearn.decomposition import PCA


def load_train():
    data = pandas.read_csv('data.csv')
    return data

def load_test():
    data = pandas.read_csv('quiz.csv')
    return data


# In[3]:

train = load_train()
test = load_test()


# In[4]:

# Name Columns
alphabet = list(string.ascii_lowercase)
alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']
train.columns = alphabet2
print(alphabet2)

# Designate Boolean Columns
boolean_cols = [
    'g', 'p', 'q', 's',
    'v', 'w', 'y', 'z',
    'oo', 'pp', 'qq', 'rr',
    'xx', 'yy', 'zz'
]

factor_cols = [
    
]


# In[10]:

# Designate Categorical Columns
cols = train.columns
num_cols = train._get_numeric_data().columns
list(set(cols) - set(num_cols))
print(num_cols)

categorical_cols = ['a', 'c', 'e', 'd', 'f',
 'uu', 'i', 'k', 'j', 'm',
 'l', 'o', 'n', 'ss', 'h',
 'tt']

# Designate Numeric Columns
numeric_cols = ['b', 'g', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
       'z', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii',
       'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp', 'qq', 'rr', 'vv',
       'ww', 'xx', 'yy', 'zz', 'aaa']


# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


model = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
print("Training residual sum of squares: %.2f" % np.mean((model.predict(train_data) - train_labels) ** 2))
print("Test residual sum of squares: %.2f" % np.mean((model.predict(test_data) - test_labels) ** 2))


clf = linear_model.Lasso(alpha=2)
clf.fit(train_data, train_labels)
Lasso(alpha=0.1, fit_intercept=True)
print(clf.coef_)
print(clf.intercept_)


h = .02  # Step size in the mesh

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)


# In[ ]:

def cross_val(cls, train_data, train_labels):
    size = math.ceil(train_data.shape[0] / 10)
    train_sets = []
    for i in range(0, 10):
        train_sets.append([train_data[i : i + size], train_labels[i : i + size]])
    
    error_rates = []
    for s in train_sets:
        # Remove the data to be used for testing
        reduced_train_set = [x for x in train_sets if x is not s]
        reduced_train_set_data = [a for l in reduced_train_set for a in l[0]]
        reduced_train_set_labels = [a for l in reduced_train_set for a in l[1]]
        classifier = cls(reduced_train_set_data, reduced_train_set_labels)
        preds = classifier.predict(s[0])
        err_rate = np.count_nonzero(preds - s[1]) / (len(s[0]) + 0.0)
        error_rates.append(err_rate)
    return np.mean(error_rates)


def meta_classify(train_data, train_labels, test_data, test_labels):
    err_rates = []
    
    classifiers = [
        AvgPerceptronClassifier,
        APCExpanded,
        LRWrapper,
        LRExpandedWrapper,
        LDAWrapper,
        QDAWrapper,
    ]
    
    categorical_cls = [
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        SVC(kernel="linear", C=0.025),
        AdaBoostClassifier(),
        SVC(gamma=2, C=1),
        GaussianNB(),
    ]

    numerical_cls = [
        KNeighborsClassifier(3),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
    ]
    
    for clf in classifiers:
        clf.fit(X_train, y_train)
        print(clf.coef_)
        score = clf.score(X_test, y_test)
    
    for cls in classifiers:
        e = cross_val(cls, train_data, train_labels)
        err_rates.append(e)
        print(cls, e)
        
    best_cls = classifiers[np.argmin(err_rates)]
    return best_cls(train_data, train_labels).predict(test_data)
            
    
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        data = load()
        preds = meta_classify(data[0], data[1], data[2], data[3])
        print(np.count_nonzero(preds - data[3]) / (len(data[3]) + 0.0))


# In[6]:

train.corr()
train.


# In[ ]:

pca = PCA(n_components=2)

