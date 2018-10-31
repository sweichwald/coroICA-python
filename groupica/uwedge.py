import itertools
import numpy as np
import scipy.linalg as lin


def uwedge(Rx,
           init=None,
           rm_x0=True,
           return_diagonals=False,
           eps=1e-10,
           n_iter_max=1000,
           minimize_loss=False,
           verbose=False,
           n_components=None):
    # Input:
    # Output:
    #   if return_diagonals=True
    #     V, diagonals, converged, iteration, meanoffdiag
    #   otherwise
    #     V, converged, iteration, meanoffdiag
    # Reference:
    # Fast Approximate Joint Diagonalization Incorporating Weight Matrices.
    # IEEE Transactions on Signal Processing, 2009.

    # 0) Preprocessing

    # Remove and remember 0st matrix
    Rx0 = np.copy(Rx[0, :, :])
    if rm_x0:
        Rx = Rx[1:, :, :]
    M, d = Rx.shape[0:2]

    if n_components is None:
        n_components = d

    # Initial guess
    if init is None and n_components == d:
        if Rx.shape[0] > 0:
            E, H = lin.eigh(Rx[0, :, :])
            V = np.dot(np.diag(1. / np.sqrt(np.abs(E))), H.T)
        else:
            V = np.eye(d)
    elif init is None:
        E, H = lin.eigh(Rx[0, :, :])
        mat = np.hstack([np.diag(1. / np.sqrt(np.abs(E[:n_components]))),
                         np.zeros((n_components, d - n_components))])
        V = np.dot(mat, H.T)
    else:
        V = init[:n_components, :]

    V = V / lin.norm(V, axis=1)[:, None]

    current_best = [None, np.inf, 0, None]

    for iteration in itertools.count():
        # 1) Generate Rs
        Rs = np.stack([V.dot(Rxx.dot(V.T)) for Rxx in Rx])

        # 2) Use Rs to construct A, equation (24) in paper with W=Id
        # 3) Set A1=Id and substitute off-diagonals
        Rsdiag = Rs.diagonal(axis1=1, axis2=2)
        Rsdiagprod = Rsdiag.T.dot(Rsdiag)
        denom_mat = np.outer(
            Rsdiagprod.diagonal(),
            Rsdiagprod.diagonal()) - Rsdiagprod**2
        Rkl = np.einsum('ill,ikl->kl', Rs, Rs)
        num_mat = Rsdiagprod.diagonal()[:, None] * Rkl - Rsdiagprod * Rkl.T
        denom_mat[denom_mat == 0] = np.finfo(V.dtype).eps
        A = num_mat / (denom_mat + np.eye(n_components))
        np.fill_diagonal(A, 1)

        # 4) Set new V
        Vold = np.copy(V)
        V = lin.lstsq(A, Vold,
                      check_finite=False,
                      lapack_driver='gelsy')[0]

        # 5) Normalise V
        V = V / lin.norm(V, axis=1)[:, None]

        if minimize_loss:
            normaliser = np.diag(V.dot(Rx0.dot(V.T)))
            Vnorm = V / (
                np.sign(normaliser) * np.sqrt(np.abs(normaliser)))[:, None]
            diagonals = np.stack([Vnorm.dot(Rxx.dot(Vnorm.T)) for Rxx in Rx])
            meanoffdiag = np.mean(
                diagonals[:, ~np.eye(n_components, dtype=bool)]**2)
            if meanoffdiag < current_best[1]:
                current_best = [Vnorm, meanoffdiag, iteration, diagonals]

        # 6) Check convergence
        changeinV = np.max(np.abs(V - Vold))
        if iteration >= n_iter_max - 1:
            converged = False
            break
        elif changeinV < eps:
            converged = True
            break

    # Rescale
    if minimize_loss:
        V, meanoffdiag, iteration, diagonals = current_best
    else:
        normaliser = np.diag(V.dot(Rx0.dot(V.T)))
        V = V / (np.sign(normaliser) * np.sqrt(np.abs(normaliser)))[:, None]
        diagonals = np.stack([V.dot(Rxx.dot(V.T)) for Rxx in Rx])
        meanoffdiag = np.mean(
            diagonals[:, ~np.eye(n_components, dtype=bool)]**2)

    # Return
    if return_diagonals:
        return V, diagonals, converged, iteration, meanoffdiag
    return V, converged, iteration, meanoffdiag
