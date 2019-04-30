from coroica.utils import md_index
from coroica.uwedge import uwedge
import numpy as np


def orthogonal_A(n, dtype='real'):
    N = np.random.normal(size=(n, n))
    if dtype == 'complex':
        N = N + 1j*np.random.normal(size=(n, n))
    A, _ = np.linalg.qr(N)
    return A


def sim_samples(d, M, dtype):
    A = orthogonal_A(d, dtype=dtype)
    V = np.linalg.inv(A)
    diags = [np.diag(np.random.uniform(1, 2, size=d))
             for i in range(M)]
    Rxx = np.stack([A.dot(D.dot(A.T.conj())) for D in diags])
    return A, V, diags, Rxx


def test_uwedge():
    for d in [3, 20]:
        for M in [3, 10]:
            for dtype in ['real', 'complex']:
                A, V, diags, Rxx = sim_samples(d, M, dtype)
                V_, converged, iteration, meanoffdiag = uwedge(Rxx)
                assert converged
                np.testing.assert_almost_equal(md_index(A, V_), 0)
