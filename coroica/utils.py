import numpy as np


def autocov(X, lag=0):
    if lag == 0:
        return np.cov(X)
    else:
        A = X[:, lag:] - X[:, lag:].mean(axis=1, keepdims=True)
        B = X[:, :-lag] - X[:, :-lag].mean(axis=1, keepdims=True)
        n = A.shape[1] - 1
        return A.dot(B.T) / n


def rigidgroup(length, nosamples):
    groups = int(np.floor(length / nosamples))
    changepoints = [int(np.round(a))
                    for a in np.linspace(0, length, groups + 1)]
    changepoints = list(set(changepoints))
    changepoints.sort()
    index = np.zeros(length)
    for (i, a, b) in zip(range(groups), changepoints[:-1], changepoints[1:]):
        index[a:b] = i
    return index
