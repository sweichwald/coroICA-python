from groupica import GroupICA
from sklearn.utils.estimator_checks import check_estimator


def test_transformer():
    return check_estimator(GroupICA)
