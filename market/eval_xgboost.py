#!/usr/bin/env python

# 0.587524024679
# 0.0117820203669

import csv
import sys

import numpy as np
import pandas as pd
import xgboost as xg

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

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    err = kappa(labels, transform_regression(preds), weights="quadratic")
    return 'error', err


def train_model(train, folds):
    y = train.median_relevance.values
    x = train.drop(["median_relevance", "doc_id"], 1).values

    xg_params = {
        "silent": 1,
        "objective": "reg:linear",
        "nthread": 4,
        "bst:max_depth": 10,
        "bst:eta": 0.1,
        "bst:subsample": 0.5
    }
    num_round = 600

    scores = []
    for train_index, test_index in cross_validation.StratifiedKFold(
            y=y,
            n_folds=int(folds),
            shuffle=True,
            random_state=42):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        xg_train = xg.DMatrix(x_train, label=y_train)
        xg_test  = xg.DMatrix(x_test,  label=y_test)

        watchlist = [(xg_train, "train"), (xg_test, "test")]
        bst = xg.train(xg_params, xg_train, num_round, watchlist, feval=evalerror)

        predicted = transform_regression(bst.predict(xg_test))

        s = kappa(y_test, predicted, weights="quadratic")
        print s
        scores.append(s)

    warn("cv scores:")
    warn(scores)
    warn(np.mean(scores))
    warn(np.std(scores))


def main():
    try:
        train_filename, features_filename, folds = sys.argv[1:]
    except:
        print("train_filename, features_filename, folds = sys.argv[1:]")
        return
    train_data = pd.read_csv(train_filename, index_col=None)
    test_data = pd.read_csv('data/test.csv', index_col=None)
    features = pd.read_csv(features_filename, index_col=None)
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
