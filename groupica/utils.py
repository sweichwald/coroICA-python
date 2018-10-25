import numpy as np


def autocov(X, lag=0):
    if lag == 0:
        return np.cov(X)
    else:
        A = X[:, lag:] - X[:, lag:].mean(axis=1, keepdims=True)
        B = X[:, :-lag] - X[:, :-lag].mean(axis=1, keepdims=True)
        n = A.shape[1] - 1
        return A.dot(B.T) / n
