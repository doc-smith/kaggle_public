from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, SVR

def lr():
    return LogisticRegression(C=3.0)

def linear_svc():
    return LinearSVC(C=0.6)

def linear_svr():
    return SVR(C=0.6, kernel="linear")


def svc():
    return SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

def svr():
    return SVR(C=7.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)

def svc5():
    return SVC(C=5.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

def svr5():
    return SVR(C=5.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)

models= {
    0  : lr,
    1  : linear_svc,
    2  : lr,
    3  : linear_svc,
    4  : svc,
    5  : svc,
    6  : svc,
    7  : svc,
    8  : svc,
    9  : svr,
    10 : svc,
    11 : svr,
    12 : svc5,
    13 : svr5,
    14 : svc5,
    15 : svr5
}
