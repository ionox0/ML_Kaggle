
# coding: utf-8

# In[8]:

import math
import pandas
import numpy as np
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.linear_model import Lasso


def load():
    data = pandas.io.parsers.read_csv('data.csv')
    return data
    
def ols(data):
    train_data = np.array(data[0:52]).astype('float')
    train_labels = np.array(data['label'])
    
    model = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    model.fit(train_data, train_labels)
    
    print(model.coef_)
    print("Training residual sum of squares: %.2f" % np.mean((model.predict(train_data) - train_labels) ** 2))
    print("Test residual sum of squares: %.2f" % np.mean((model.predict(test_data) - test_labels) ** 2))

def lasso(data):
    train_data = np.array(data['data']).astype('float')
    train_labels = np.array(data['labels'])
    
    clf = linear_model.Lasso(alpha=2)
    clf.fit(train_data, train_labels)
    Lasso(alpha=0.1, fit_intercept=True)
    
    print(clf.coef_)
    print(clf.intercept_)
    


# In[9]:

data = load()


# In[10]:

ols(data)


# In[ ]:



