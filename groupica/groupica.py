"""
groupICA
"""
from .uwedge import uwedge
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


class GroupICA(BaseEstimator, TransformerMixin):
    """groupICA transformer

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
        When true, the components will be ordered in decreasing signal
        strengths.
    pairing : {'complement', 'allpairs'}
        Whether difference matrices should be computed for all pairs of
        partition covariance matrices or only in a one-vs-complement scheme.
    groupsize : int, optional
        Approximately how many samples, when doing a rigid grid, shall be in
        each group. If none is passed, all samples will be in one group unless
        group_index is passed during fitting in which case the provided group
        index is used (the latter is the adviced and preferred way).
    partitionsize : int, optional
        Approxiately how many samples, when doing a rigid grid, should be in
        each partition. If none is passed, a (hopefully sane) default is used,
        again, unless partition_index is passed during fitting in which case
        the provided partition index is used.
    max_iter : int, optional
        Maximum number of iterations for the uwedge approximate joint
        diagonalisation during fitting.
    tol : float, optional
        Tolerance for terminating the uwedge approximate joint diagonalisation
        during fitting.

    Attributes
    ----------
    V_ : array, shape (n, n_features)
        The unmixing matrix; where n=n_features if n_components_uwedge is None
        and n=n_components_uwedge otherwise. n_components will be taken into
        account during transform only; the unmixing matrix is kept complete.
    converged_ : boolean
        Whether the approximate joint diagonalisation converged due to tol.
    n_iter_ : int
        Number of iterations of the approximate joint diagonalisation.
    meanoffdiag_ : float
        Mean absolute value of the off-diagonal values of the to be jointly
        diagonalised matrices, i.e., a proxy of the approximate joint
        diagonalisation objective function.
    sig2noise_ : float
        A measure of signal strenghts in the difference matrices; used for
        internal review only.
    """

    def __init__(self,
                 n_components=None,
                 n_components_uwedge=None,
                 rank_components=False,
                 pairing='complement',
                 groupsize=None,
                 partitionsize=None,
                 max_iter=1000,
                 tol=1e-12):
        self.n_components = n_components
        self.n_components_uwedge = n_components_uwedge
        self.rank_components = rank_components
        self.pairing = pairing
        self.groupsize = groupsize
        self.partitionsize = partitionsize
        self.max_iter = max_iter
        self.tol = tol

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
            is provided a rigid grid with self.groupsize_ samples per
            group is used (which defaults to all samples if self.groupsize_
            was not set).
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
        X = check_array(X)

        n, dim = X.shape

        # generate group and partition indices as needed
        if group_index is None and self.groupsize is None:
            group_index = np.zeros(n)
        elif group_index is None:
            group_index = rigidgroup(n, self.groupsize)
        if partition_index is None and self.partitionsize is None:
            smallest_group = np.min([(group_index == group).sum()
                                     for group in np.unique(group_index)])
            partition_index = rigidpartition(
                group_index,
                np.max([dim, smallest_group // 2]))
        elif partition_index is None:
            partition_index = rigidpartition(group_index, self.partitionsize)

        X, group_index = check_X_y(X, group_index)
        X, partition_index = check_X_y(X, partition_index)

        X = X.T

        no_groups = np.unique(group_index).shape[0]

        # computing covariance difference matrices
        if self.pairing == 'complement':
            no_pairs = 0
            for group in np.unique(group_index):
                no_pairs += len(
                    np.unique(partition_index[group_index == group]))
            covmats = np.zeros((no_pairs, dim, dim))
            idx = 0
            for group in np.unique(group_index):
                for partition in np.unique(
                        partition_index[group_index == group]):
                    ind1 = ((partition_index == partition) &
                            (group_index == group))
                    ind2 = ((partition_index != partition) &
                            (group_index == group))
                    covmats[idx, :, :] = np.cov(X[:, ind1]) - \
                        np.cov(X[:, ind2])
                    idx += 1
        elif self.pairing == 'allpairs':
            no_pairs = 0
            subvec = np.zeros(no_groups, dtype=int)
            for i, group in enumerate(np.unique(group_index)):
                subvec[i] = np.int(len(
                    np.unique(partition_index[group_index == group])))
                no_pairs += int(subvec[i] * (subvec[i] - 1) / 2)
            covmats = np.zeros((no_pairs, dim, dim))
            idx = 0
            for count, group in enumerate(np.unique(group_index)):
                unique_subs = np.unique(partition_index[group_index == group])
                for i in range(subvec[count] - 1):
                    for j in range(i + 1, subvec[count]):
                        ind1 = ((partition_index == unique_subs[i]) &
                                (group_index == group))
                        ind2 = ((partition_index == unique_subs[j]) &
                                (group_index == group))
                        covmats[idx, :, :] = np.cov(X[:, ind1]) - \
                            np.cov(X[:, ind2])
                        idx += 1

        # add total observational covariance for normalization
        covmats = np.concatenate((np.cov(X)[None, ...], covmats), axis=0)

        # joint diagonalisation
        self.V_, self.converged_, self.n_iter_, self.meanoffdiag_, \
            self.sig2noise_ = uwedge(
                covmats,
                rm_x0=True,
                eps=self.tol,
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
        if self.n_components is None:
            return self.V_.dot(X.T).T
        else:
            return self.V_[:self.n_components, :].dot(X.T).T


def rigidpartition(group, nosamples):
    partition = np.zeros(group.shape)
    for e in np.unique(group):
        partition[np.where(group == e)] = rigidgroup(
            (group == e).sum(),
            nosamples) + partition.max() + 1
    return partition


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
