from __future__ import division
import warnings
import string
import argparse

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument("datafile")
parser.add_argument("quizfile")
parser.add_argument("outputfile")
args = parser.parse_args()

# Create useful column subests
paired_cols = [
    ['a', 'f', 'k'],
    ['b', 'g', 'l'],
    ['c', 'h', 'm'],
    ['d', 'i', 'n'],
    ['e', 'j', 'o'],
    ['ss', 'tt', 'uu']
]

categorical_cols = [
    'a', 'c', 'd', 'e',
    'f', 'h', 'i', 'j',
    'k', 'l', 'n', 'm',
    'o', 'ss', 'tt', 'uu'
]

three_value_cols = [
    'aa', 'bb', 'cc',
    'dd', 'ee', 'ff',
    'gg', 'hh', 'ii',
    'jj', 'kk', 'll',
    'mm', 'nn'
]

boolean_cols = [
    'p', 'q',  's',
    'v', 'w',
    'y', 'z', 'oo',
    'pp', 'qq', 'rr',
    'xx', 'yy', 'zz'
] # 't', 'u', 'r', 'x',

potentially_useless_cols = [
    'x', 'u', 't', 'r'
]

continuous_cols = [
    'vv', 'ww'
]


def score_features(clf, X, y, n_iterations=1):
    rankings = []
    for it in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf.fit(X, y)
        rankings.append(clf.feature_importances_)
    return np.array(rankings)


def main():
    DATAFILE = args.datafile
    QUIZFILE = args.quizfile
    OUTPUTFILE = args.outputfile

    # Read in the data
    task = pd.read_csv(DATAFILE)
    quiz = pd.read_csv(QUIZFILE)

    # Rename columns to make the easier to work with
    alphabet = list(string.ascii_lowercase)
    alphabet2 = alphabet + [l+l for l in alphabet] + ['aaa']

    task.columns = alphabet2
    quiz.columns = alphabet2[:-1]

    # One-hot encode the columns
    final_cols = categorical_cols + three_value_cols + boolean_cols + continuous_cols
    X_dummies = pd.get_dummies(task[final_cols])
    X_quiz_dummies = pd.get_dummies(quiz[final_cols])
    X_train_dummies = X_dummies[[col for col in X_dummies.columns if col in X_quiz_dummies.columns]]
    X_quiz_dummies = X_quiz_dummies[[col for col in X_quiz_dummies.columns if col in X_train_dummies.columns]]
    X_train_sample = X_train_dummies.sample(10000)
    target_cols = [col for col in X_dummies.columns if col in X_quiz_dummies.columns]
    sample_ixs = list(X_train_sample.index)
    y = task.aaa.as_matrix()
    y_sample = y[sample_ixs]

    # Feature selection using the model
    rf_clf = RandomForestClassifier(n_estimators=30, n_jobs=-1)
    feats = score_features(rf_clf, X_train_sample.as_matrix(), y_sample, n_iterations=20)
    imp_sums = pd.Series(feats.sum(axis=0))
    gt_zero_features = imp_sums[imp_sums > 0].index
    gt_zero_cols = X_train_dummies.columns[gt_zero_features]

    # Fit the classifier and predict
    X = X_train_dummies[gt_zero_cols]
    rf_clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    rf_clf.fit(X, y)
    preds = rf_clf.predict(X_quiz_dummies[gt_zero_cols].as_matrix())
    pred = pd.DataFrame(preds).reset_index()
    pred.columns = ['Id', 'Prediction']
    pred.Id = pred.Id + 1
    pred.to_csv(OUTPUTFILE, index=False)


if __name__ == '__main__':
    main()


