from coroica import CoroICA, UwedgeICA
from sklearn.utils.estimator_checks import check_estimator


def test_transformer_coroICA():
    return check_estimator(CoroICA)


def test_transformer_uwedgeICA():
    return check_estimator(UwedgeICA)
