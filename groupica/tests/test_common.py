from groupica import GroupICA, UwedgeICA
from sklearn.utils.estimator_checks import check_estimator


def test_transformer_groupICA():
    return check_estimator(GroupICA)


def test_transformer_uwedgeICA():
    return check_estimator(UwedgeICA)
