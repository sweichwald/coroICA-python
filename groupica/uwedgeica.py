# coding: utf-8
"""
uwedgeICA
Implementation of the NSS uwedgeICA algorithms used for comparison in
groupICA: Independent component analysis for grouped data
N Pfister*, S Weichwald*, P Bühlmann, B Schölkopf
https://arxiv.org/abs/1806.01094
"""
from .utils import autocov
from .utils import rigidgroup as rigidpartition
from .uwedge import uwedge
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
import warnings


class UwedgeICA(BaseEstimator, TransformerMixin):
    """uwedgeICA transformer

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If none is passed, the same number of
        components as the input has dimensions is used.
    n_components_uwedge : int, optional
        Number of components to extract during uwedge approximate joint
        diagonalization of the matrices. If none is passed, the same number of
        components as the input has dimensions is used.
    rank_components : boolean, optional
        When true, the components will be ordered in decreasing stability.
    partitionsize : int, optional
        Approxiately how many samples, when doing a rigid grid, should be in
        each partition. If none is passed, a (hopefully sane) default is used
        unless partition_index is passed during fitting in which case
        the provided partition index is used.
    timelags : list of ints, optional
        List of time lags to be considered for computing lagged covariance
        matrices.
    instantcov : boolean, optional
        If False, no non-lagged instant (lag = 0) covariance matrices are used.
    max_iter : int, optional
        Maximum number of iterations for the uwedge approximate joint
        diagonalisation during fitting.
    tol : float, optional
        Tolerance for terminating the uwedge approximate joint diagonalisation
        during fitting.
    minimize_loss : boolean, optional
        If True at each iteration the loss of the uwedge approximate joint
        diagonalisation is computed (computationally expensive) and after
        convergence the V with minimal loss along the optimisation path is
        returned instead of the terminal V.

    Attributes
    ----------
    V_ : array, shape (n, n_features)
        The unmixing matrix; where n=n_features if n_components and
        n_components_uwedge are None, n=n_components_uwedge if n_components is
        None, and n=n_components otherwise.
    converged_ : boolean
        Whether the approximate joint diagonalisation converged due to tol.
    n_iter_ : int
        Number of iterations of the approximate joint diagonalisation.
    meanoffdiag_ : float
        Mean absolute value of the off-diagonal values of the to be jointly
        diagonalised matrices, i.e., a proxy of the approximate joint
        diagonalisation objective function.
    """

    def __init__(self,
                 n_components=None,
                 n_components_uwedge=None,
                 rank_components=False,
                 partitionsize=None,
                 timelags=None,
                 instantcov=True,
                 max_iter=1000,
                 tol=1e-12,
                 minimize_loss=False):
        self.n_components = n_components
        self.n_components_uwedge = n_components_uwedge
        self.rank_components = rank_components
        self.partitionsize = partitionsize
        self.timelags = timelags
        self.instantcov = instantcov
        self.max_iter = max_iter
        self.tol = tol
        self.minimize_loss = minimize_loss
        if self.timelags is None and not self.instantcov:
            warnings.warn('timelags=None and instantcov=True results in the '
                          'identity transformer, since no (lagged) covariance '
                          'matrices are to be diagonalised.',
                          UserWarning)

    def fit(self, X, y=None, partition_index=None):
        """Fit the model

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            where n_samples is the number of samples and
            n_features is the number of features.
        y : Ignored.
        partition_index : array, optional, shape (n_samples,)
            Codes for each sample which partition it belongs to; if no
            partition index is provided a rigid grid with self.partitionsize_
            samples per partition within each group is used (which has a
            (hopefully sane) default if self.partitionsize_ was not set).

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, ensure_2d=True)

        n, dim = X.shape

        if (n <= 1
                or dim <= 1
                or (self.timelags is None and not self.instantcov)):
            self.V_ = np.eye(dim)
            return self

        # generate partition index as needed
        if partition_index is None and self.partitionsize is None:
            partition_index = rigidpartition(
                n,
                np.max([dim, n // 2]))
        elif partition_index is None:
            partition_index = rigidpartition(n, self.partitionsize)

        X, partition_index = check_X_y(X, partition_index)

        X = X.T

        # computing covariance matrices
        no_partitions = len(np.unique(partition_index))
        timelags = []
        if self.instantcov:
            timelags.append(0)
        if self.timelags is not None:
            timelags.extend(self.timelags)
        no_timelags = len(timelags)
        covmats = np.empty((no_partitions * no_timelags, dim, dim))
        idx = 0
        for partition in np.unique(partition_index):
            ind = (partition_index == partition)
            for timelag in timelags:
                covmats[idx, :, :] = autocov(X[:, ind], lag=timelag)
                idx += 1

        # joint diagonalisation
        self.V_, self.converged_, self.n_iter_, self.meanoffdiag_ = uwedge(
            covmats,
            Rx0=np.cov(X),
            eps=self.tol,
            minimize_loss=self.minimize_loss,
            n_iter_max=self.max_iter,
            n_components=self.n_components_uwedge)

        # rank components
        if self.rank_components or self.n_components is not None:
            A = linalg.pinv(self.V_)
            colcorrs = np.zeros(self.V_.shape[0])
            # running average
            for k in range(covmats.shape[0]):
                colcorrs += np.abs(np.corrcoef(
                    A.T,
                    self.V_.dot(covmats[k, ...].T)
                )[:dim, dim:].diagonal() / covmats.shape[0])
            sorting = np.argsort(colcorrs)[::-1]
            self.V_ = self.V_[sorting, :]

        if self.n_components is not None:
            self.V_ = self.V_[:self.n_components, :]

        return self

    def transform(self, X):
        """Returns the data projected onto the fitted components

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
        """
        check_is_fitted(self, ['V_'])
        X = check_array(X)
        return self.V_.dot(X.T).T
