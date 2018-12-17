# coding: utf-8
"""
coroICA
Implementation of the coroICA algorithm presented in
Robustifying Independent Component Analysis
by Adjusting for Group-Wise Stationary Noise
N Pfister*, S Weichwald*, P Bühlmann, B Schölkopf
https://arxiv.org/abs/1806.01094
"""
from .utils import autocov, rigidgroup
from .uwedge import uwedge
import itertools
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.random import check_random_state
from sklearn.utils.validation import check_is_fitted
import warnings


class CoroICA(BaseEstimator, TransformerMixin):
    """coroICA transformer

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
    pairing : {'complement', 'allpairs', 'neighbouring'}
        Whether difference matrices should be computed for all pairs of
        partition covariance matrices or only in a one-vs-complement scheme or
        only of neighbouring partition covariance matrices.
    max_matrices : float or 'no_partitions', optional (default=1)
        The fraction of (lagged) covariance matrices to use during training
        or, if 'no_partitions', at most as many covariance matrices are used
        as there are partitions.
    groupsize : int, optional
        Approximately how many samples, when doing a rigid grid, shall be in
        each group. If none is passed, all samples will be in one group unless
        group_index is passed during fitting in which case the provided group
        index is used (the latter is the advised and preferred way).
    partitionsize : int or list of int, optional
        Approximately how many samples, when doing a rigid grid, should be in
        each partition. If none is passed, a (hopefully sane) default is used
        unless partition_index is passed during fitting in which case
        the provided partition index is used.
    partitionsize : int, optional
        Approximately how many samples, when doing a rigid grid, should be in
        each partition. If none is passed, a (hopefully sane) default is used,
        again, unless partition_index is passed during fitting in which case
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
    condition_threshold : int, optional (default=None)
        If int, the uwedge iteration is stopped when the condition number of
        the unmixing matrix grows beyond condition_threshold. If None, no such
        condition number check is performed.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is seeded used by the random number generator;
        if RandomState instance, random_state is the random number generator;
        if None, the random number generator is the RandomState instance used
        by np.random.

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
                 pairing='complement',
                 max_matrices=1,
                 groupsize=None,
                 partitionsize=None,
                 timelags=None,
                 instantcov=True,
                 max_iter=5000,
                 tol=1e-12,
                 minimize_loss=False,
                 condition_threshold=None,
                 random_state=None):
        self.n_components = n_components
        self.n_components_uwedge = n_components_uwedge
        self.rank_components = rank_components
        self.pairing = pairing
        self.max_matrices = max_matrices
        self.groupsize = groupsize
        self.partitionsize = partitionsize
        self.timelags = timelags
        self.instantcov = instantcov
        self.max_iter = max_iter
        self.tol = tol
        self.minimize_loss = minimize_loss
        self.condition_threshold = condition_threshold
        self.random_state = random_state
        if self.timelags is None and not self.instantcov:
            warnings.warn('timelags=None and instantcov=True results in the '
                          'identity transformer, since no (lagged) covariance '
                          'matrices are to be diagonalised.',
                          UserWarning)

    def fit(self, X, y=None, group_index=None, partition_index=None):
        """Fit the model

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            where n_samples is the number of samples and
            n_features is the number of features.
        y : Ignored.
        group_index : array, optional, shape (n_samples,)
            Codes for each sample which group it belongs to; if no group index
            is provided a rigid grid with self.groupsize samples per
            group is used (which defaults to all samples if self.groupsize
            was not set).
        partition_index : array, optional, shape (n_samples,)
            Codes for each sample which partition it belongs to; if no
            partition index is provided a rigid grid with self.partitionsize
            samples per partition within each group is used (which has a
            (hopefully sane) default if self.partitionsize was not set).

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, ensure_2d=True)

        n, dim = X.shape

        if n <= 1 or dim <= 1:
            self.V_ = np.eye(dim)
            return self

        random_state = check_random_state(self.random_state)

        # generate group and partition indices as needed
        if group_index is None and self.groupsize is None:
            group_index = np.zeros(n)
        elif group_index is None:
            group_index = rigidgroup(n, self.groupsize)

        # generate partition index as needed
        if partition_index is None and self.partitionsize is None:
            smallest_group = np.min([(group_index == group).sum()
                                     for group in np.unique(group_index)])
            partition_indices = [rigidpartition(
                group_index,
                np.max([dim, smallest_group // 2]))]
        elif partition_index is None and type(self.partitionsize) == list:
            partition_indices = [rigidpartition(group_index, partsize)
                                 for partsize in self.partitionsize]
        elif partition_index is None:
            partition_indices = [
                rigidpartition(group_index, self.partitionsize)]
        else:
            partition_indices = [partition_index]

        X, group_index = check_X_y(X, group_index)
        for partition_index in partition_indices:
            X, partition_index = check_X_y(X, partition_index)

        X = X.T

        no_groups = np.unique(group_index).shape[0]

        # computing covariance difference matrices
        timelags = []
        if self.instantcov:
            timelags.append(0)
        if self.timelags is not None:
            timelags.extend(self.timelags)
        no_timelags = len(timelags)
        for partition_index in partition_indices:
            if self.pairing == 'complement':
                if self.max_matrices == 'no_partitions':
                    max_matrices = 1.0
                else:
                    max_matrices = self.max_matrices
                no_pairs = 0
                for group in np.unique(group_index):
                    no_pairs += len(
                        np.unique(partition_index[group_index == group]))
                covmats = np.empty((no_pairs * no_timelags, dim, dim))
                idx = 0
                for group in np.unique(group_index):
                    unique_partitions = np.unique(
                        partition_index[group_index == group])
                    unique_partitions = random_state.choice(
                        unique_partitions,
                        size=np.ceil(
                            max_matrices * unique_partitions.shape[0]
                        ).astype(int),
                        replace=False)
                    for partition in unique_partitions:
                        ind1 = ((partition_index == partition) &
                                (group_index == group))
                        ind2 = ((partition_index != partition) &
                                (group_index == group))
                        if ind2.sum() > 0:
                            for timelag in timelags:
                                covmats[idx, :, :] = autocov(
                                    X[:, ind1], lag=timelag) - \
                                    autocov(X[:, ind2], lag=timelag)
                                idx += 1
                        else:
                            warnings.warn('Removing group {} since the '
                                          'partition is trivial, i.e., '
                                          'contains only exactly one '
                                          'set'.format(group),
                                          UserWarning)
            elif self.pairing in ['allpairs', 'neighbouring']:
                no_pairs = 0
                pairs_per_group = [None] * no_groups
                for i, group in enumerate(np.unique(group_index)):
                    unique_partitions = np.unique(
                        partition_index[group_index == group])
                    if self.pairing == 'allpairs':
                        pairs = np.asarray(list(
                            itertools.combinations(unique_partitions, 2)))
                    elif self.pairing == 'neighbouring':
                        pairs = np.asarray([
                            [unique_partitions[k], unique_partitions[k + 1]]
                            for k in range(len(unique_partitions) - 1)])
                    if pairs.shape[0] == 0:
                            warnings.warn('Removing group {} since the '
                                          'partition is trivial, i.e., '
                                          'contains only exactly one '
                                          'set'.format(group),
                                          UserWarning)
                    else:
                        if self.max_matrices == 'no_partitions':
                            max_matrices = (len(unique_partitions) - 1) / \
                                pairs.shape[0]
                        else:
                            max_matrices = self.max_matrices
                        pairs_per_group[i] = pairs[random_state.choice(
                            pairs.shape[0],
                            size=np.ceil(
                                max_matrices * pairs.shape[0]
                            ).astype(int),
                            replace=False
                        )]
                        no_pairs += pairs_per_group[i].shape[0]
                covmats = np.empty((no_pairs * no_timelags, dim, dim))
                idx = 0
                for pairs, group in zip(
                        pairs_per_group, np.unique(group_index)):
                    if pairs is not None:
                        for i, j in pairs:
                            ind1 = ((partition_index == i) &
                                    (group_index == group))
                            ind2 = ((partition_index == j) &
                                    (group_index == group))
                            for timelag in timelags:
                                covmats[idx, :, :] = autocov(
                                    X[:, ind1], lag=timelag) - \
                                    autocov(X[:, ind2], lag=timelag)
                                idx += 1
        covmats = covmats[:idx, :, :]

        Rx0 = np.cov(X)

        # joint diagonalisation
        self.V_, self.converged_, self.n_iter_, self.meanoffdiag_ = uwedge(
            covmats,
            Rx0=Rx0,
            eps=self.tol,
            minimize_loss=self.minimize_loss,
            n_iter_max=self.max_iter,
            n_components=self.n_components_uwedge,
            condition_threshold=self.condition_threshold)

        # normalise V
        normaliser = np.diag(self.V_.dot(Rx0.dot(self.V_.T)))
        self.V_ = self.V_ / (
            np.sign(normaliser) * np.sqrt(np.abs(normaliser)))[:, None]

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


def rigidpartition(group, nosamples):
    partition = np.zeros(group.shape)
    for e in np.unique(group):
        partition[np.where(group == e)] = rigidgroup(
            (group == e).sum(),
            nosamples) + partition.max() + 1
    return partition
