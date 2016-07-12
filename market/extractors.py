import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

def t_svd_q(features):
    x_train = csr_matrix(hstack((
                features.title_tf_idf_train,
              )))

    x_test = csr_matrix(hstack((
                features.title_tf_idf_test,
              )))

    svd = TruncatedSVD(n_components=250, n_iter=5)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

    x_train = np.hstack((
        x_train,
        features.queries_tf_idf_train.toarray(),
        features.features_train
        ))
    x_test = np.hstack((
        x_test,
        features.queries_tf_idf_test.toarray(),
        features.features_test
        ))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train


def secret_1(features):
    x_title = csr_matrix(hstack((
                features.title_tf_idf_train,
              )))

    x_description = csr_matrix(hstack((
                features.description_tf_idf_train,
              )))

    x_title_test = csr_matrix(hstack((
                features.title_tf_idf_test,
              )))

    x_description_test = csr_matrix(hstack((
                features.description_tf_idf_test,
              )))

    title_svd = TruncatedSVD(n_components=250, n_iter=5)
    title_svd.fit(x_title)

    description_svd = TruncatedSVD(n_components=30, n_iter=5)
    description_svd.fit(x_description)

    x_train = np.hstack((
        title_svd.transform(x_title),
        description_svd.transform(x_description),
        features.queries_tf_idf_train.toarray(),
        features.features_train
        ))
    x_test = np.hstack((
        title_svd.transform(x_title_test),
        description_svd.transform(x_description_test),
        features.queries_tf_idf_test.toarray(),
        features.features_test
        ))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train


def td_svd_q(features):
    x_train = csr_matrix(hstack((
                features.title_tf_idf_train,
                features.description_tf_idf_train
              )))

    x_test = csr_matrix(hstack((
                features.title_tf_idf_test,
                features.description_tf_idf_test
              )))

    svd = TruncatedSVD(n_components=250, n_iter=5)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

    x_train = np.hstack((
        x_train,
        features.queries_tf_idf_train.toarray(),
        features.features_train
        ))
    x_test = np.hstack((
        x_test,
        features.queries_tf_idf_test.toarray(),
        features.features_test
        ))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train

def qtd_svd(features):
    x_train = csr_matrix(hstack((
                features.queries_tf_idf_train,
                features.title_tf_idf_train,
                features.description_tf_idf_train
              )))

    x_test = csr_matrix(hstack((
                features.queries_tf_idf_test,
                features.title_tf_idf_test,
                features.description_tf_idf_test
              )))

    svd = TruncatedSVD(n_components=200, n_iter=5)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

    x_train = np.hstack((x_train, features.features_train))
    x_test = np.hstack((x_test, features.features_test))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train

def qt_svd(features):
    x_train = csr_matrix(hstack((
                features.queries_tf_idf_train,
                features.title_tf_idf_train
              )))

    x_test = csr_matrix(hstack((
                features.queries_tf_idf_test,
                features.title_tf_idf_test
              )))

    svd = TruncatedSVD(n_components=200, n_iter=5)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

    x_train = np.hstack((x_train, features.features_train))
    x_test = np.hstack((x_test, features.features_test))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train



def qt_tf_idf(features):
    x_train = csr_matrix(hstack((
                features.queries_tf_idf_train,
                features.title_tf_idf_train,
                features.features_train
              )))

    x_test = csr_matrix(hstack((
                features.queries_tf_idf_test,
                features.title_tf_idf_test,
                features.features_test
              )))

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train

def qtd_tf_idf(features):
    x_train = csr_matrix(hstack((
                features.queries_tf_idf_train,
                features.title_tf_idf_train,
                features.description_tf_idf_train,
                features.features_train
              )))

    x_test = csr_matrix(hstack((
                features.queries_tf_idf_test,
                features.title_tf_idf_test,
                features.description_tf_idf_test,
                features.features_test
              )))
    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train


def td_svd_dummies(features):
    x_train = csr_matrix(hstack((
                features.description_tf_idf_train,
                features.title_tf_idf_train
              )))

    x_test = csr_matrix(hstack((
                features.description_tf_idf_test,
                features.title_tf_idf_test
              )))

    svd = TruncatedSVD(n_components=200, n_iter=5)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

    x_train = np.hstack((
        x_train,
        features.features_train,
        features.train_query_dummies
    ))
    x_test = np.hstack((
        x_test,
        features.features_test,
        features.test_query_dummies
    ))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train


def t_svd_dummies(features):
    x_train = csr_matrix(hstack((
                features.title_tf_idf_train,
              )))

    x_test = csr_matrix(hstack((
                features.title_tf_idf_test,
              )))

    svd = TruncatedSVD(n_components=200, n_iter=5)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

    x_train = np.hstack((
        x_train,
        features.features_train,
        features.train_query_dummies
    ))
    x_test = np.hstack((
        x_test,
        features.features_test,
        features.test_query_dummies
    ))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = features.train['median_relevance'].values

    return x_train, x_test, y_train


extractors = {
    0  : qt_tf_idf,
    1  : qt_tf_idf,
    2  : qtd_tf_idf,
    3  : qtd_tf_idf,
    4  : qtd_svd,
    5  : qt_svd,
    6  : td_svd_q,
    7  : t_svd_q,
    8  : td_svd_dummies,
    9  : td_svd_dummies,
    10 : t_svd_dummies,
    11 : t_svd_dummies,
    12 : td_svd_dummies,
    13 : td_svd_dummies,
    14 : t_svd_dummies,
    15 : t_svd_dummies
}
