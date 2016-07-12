from scipy.sparse import csr_matrix 
import numpy as np

def transform_regression(scores):
    return map(
        lambda p: min(max(1, int(round(p))), 4),
        scores
    )

def save_csr(filename, array):
    np.savez(filename,
             data = array.data,
             indices=array.indices,
             indptr =array.indptr,
             shape=array.shape)

def load_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'],
                       loader['indices'],
                       loader['indptr']),
                      shape = loader['shape'])
