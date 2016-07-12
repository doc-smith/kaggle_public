from __future__ import division
import csv
import sys
import pandas as pd
import numpy as np

from util import load_csr, transform_regression as transform
from extractors import extractors
from models import models
from sklearn.cross_validation import StratifiedKFold
from skll.metrics import kappa
from multiprocessing import Pool
from itertools import repeat


class TFeatures:

    def __init__(self):
        self.__load_test_and_train()
        self.__load_tf_idf()
        self.__load_features()
        self.__create_dummy_variables_for_queries()

    def __load_features(self):
        features = pd.read_csv('tmp/features.csv', index_col=None)

        train = pd.merge(features,
                     self.train[["id", "median_relevance"]],
                     left_on="doc_id",
                     right_on="id").drop(["id", "doc_id", "median_relevance"], 1)

        self.features_train = train.values

        test = pd.merge(features,
                     self.test[["id"]],
                     left_on="doc_id",
                     right_on="id").drop(["id", "doc_id"], 1)
        self.features_test = test.values

    def __load_test_and_train(self):
        self.train = pd.read_csv('data/train.csv', index_col=None)
        self.test = pd.read_csv('data/test.csv', index_col=None)

    def __load_tf_idf(self):
        self.description_tf_idf_train = load_csr('tmp/description_tf_idf_train.npz')
        self.description_tf_idf_test = load_csr('tmp/description_tf_idf_test.npz')
        self.queries_tf_idf_train = load_csr('tmp/queries_tf_idf_train.npz')
        self.queries_tf_idf_test = load_csr('tmp/queries_tf_idf_test.npz')
        self.title_description_tf_idf_train = load_csr('tmp/title_description_tf_idf_train.npz')
        self.title_description_tf_idf_test = load_csr('tmp/title_description_tf_idf_test.npz')
        self.title_tf_idf_train = load_csr('tmp/titles_tf_idf_train.npz')
        self.title_tf_idf_test = load_csr('tmp/titles_tf_idf_test.npz')

    def __create_dummy_variables_for_queries(self):
        self.train_query_dummies = pd.get_dummies(self.train["query"])
        self.test_query_dummies = pd.get_dummies(self.test["query"])


class TModel:

    def __init__(self, features, model):
        self.folds = 10
        self.features = features
        self.model = model
        self.__build_inputs()
        self.__build_model()

    def __build_inputs(self):
        sys.stderr.write('Building inputs\n')
        self.x_train, self.x_test, self.y_train = extractors[self.model](self.features)

    def __build_model(self):
        sys.stderr.write('Building model\n')
        self.clf = models[self.model]()

    def fit(self, train_index):
        sys.stderr.write('Fit {0}\n'.format(self.model))
        x_train = self.x_train[train_index]
        y_train = self.y_train[train_index]
        self.clf.fit(x_train, y_train)
        return self.clf

    def predict(self, test_index):
        sys.stderr.write('Predict {0}\n'.format(self.model))
        x_test = self.x_train[test_index]
        y_test = self.y_train[test_index]
        return self.clf.predict(x_test), y_test

    def eval(self):
        sys.stderr.write('Evaluating\n')
        folds = StratifiedKFold(y=self.y_train, n_folds=self.folds, shuffle=True, random_state=1337)
        scores = []
        for train_index, test_index in folds:
            self.fit(train_index)
            predicted, y_test = self.predict(test_index)
            k = kappa(y_test, transform(predicted), weights='quadratic')
            print(k)
            scores.append(k)
        print(scores)
        print(np.mean(scores))
        print(np.std(scores))

    def solution(self):
        sys.stderr.write("Solution {0}\n".format(self.model))
        self.clf.fit(self.x_train, self.y_train)
        return self.clf.predict(self.x_test)

def train_model(train_index_and_model):
    train_index, model = train_index_and_model
    model.fit(train_index)
    return model

def solution(model):
    return model.solution()

class TEnsemble:

    def __init__(self, features, models):
        self.folds = 10
        self.features = features
        self.pool = Pool(10)
        self.models = [TModel(self.features, model) for model in models]

    def fit(self, train_index):
        self.models = self.pool.map(train_model, zip(repeat(train_index), self.models))

    def predict(self, test_index):
        predicts, y_tests = zip(*[model.predict(test_index) for model in self.models])
        return list(predicts), y_tests[0]

    def eval(self):
        sys.stderr.write('Evaluating\n')
        meta_folds = StratifiedKFold(y=self.models[0].y_train, n_folds=self.folds, shuffle=True, random_state=1337)
        scores = []
        for train_index, test_index in meta_folds:
            self.fit(train_index)
            predicts, y_test = self.predict(test_index)
            predicted = [elem / len(self.models) for elem in sum(predicts)]

            k = kappa(y_test, transform(predicted), weights='quadratic')
            print(k)
            scores.append(k)
        print(scores)
        print(np.mean(scores))
        print(np.std(scores))

    def solution(self):
        predicts = self.pool.map(solution, self.models)
        predicted = [elem / len(self.models) for elem in sum(predicts)]
        return transform(predicted)


features = TFeatures()

ensemble = TEnsemble(features, [8, 9, 10, 11, 12, 13, 14, 15])
ensemble.eval()

solution = ensemble.solution()

test = pd.read_csv('data/test.csv', index_col=None)
test["prediction"] = solution
test[["id", "prediction"]].to_csv("solution.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
