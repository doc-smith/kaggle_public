#!/usr/bin/env python

# 0.643101301291
# 0.0118400202286

import csv
import sys

import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn import ensemble
from sklearn import metrics
from skll.metrics import kappa
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def warn(msg):
    print >>sys.stderr, msg


def transform_regression(scores):
    return map(
        lambda p: min(max(1, int(round(p))), 4),
        scores
    )


def train_model(train, folds):
    y = train.median_relevance.values
    x = train.drop(["median_relevance", "doc_id"], 1).values

    clf = Pipeline([
        ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
        ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))
        ])


    scores = []
    for train_index, test_index in cross_validation.StratifiedKFold(
            y=y,
            n_folds=int(folds),
            shuffle=True,
            random_state=42):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(x_train, y_train)
        predicted = transform_regression(clf.predict(x_test))

        s = kappa(y_test, predicted, weights="quadratic")
        print s
        scores.append(s)

    warn("cv scores:")
    warn(scores)
    warn(np.mean(scores))
    warn(np.std(scores))

    clf.fit(x, y)

    return clf

def main():
    folds = sys.argv[1]
    train_data = pd.read_csv('data/train.csv', index_col=None)
    test_data = pd.read_csv('data/test.csv', index_col=None)
    features = pd.read_csv('tmp/features.csv', index_col=None)
    train_svd = pd.read_csv('tmp/train_svd.csv', index_col=None)
    test_svd = pd.read_csv('tmp/test_svd.csv', index_col=None)

    train = pd.merge(features,
                     train_data[["id", "median_relevance"]],
                     left_on="doc_id",
                     right_on="id").drop("id", 1)

    test = pd.merge(features,
                     test_data[["id"]],
                     left_on="doc_id",
                     right_on="id").drop("id", 1)

    train = train.join(train_svd)
    test = test.join(test_svd)

    model = train_model(train, folds)

    test["prediction"] = transform_regression(
        model.predict(test.drop("doc_id", 1))
    )
    test[["doc_id", "prediction"]].rename(columns={
        "doc_id": "id"
    }).to_csv("solution.csv",
              index=False,
              quoting=csv.QUOTE_NONNUMERIC)


    train["prediction"] = transform_regression(model.predict(train.drop(["doc_id", "median_relevance"], 1)))
    train.to_csv("debug.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)


    return 0


if __name__ == "__main__":
    sys.exit(main())
