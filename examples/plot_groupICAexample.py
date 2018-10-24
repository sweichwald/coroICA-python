"""
====================
Minimalistic example
====================

Demonstrate basic use of :class:`groupica.GroupICA`
"""
import numpy as np
from groupica import GroupICA, UwedgeICA
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

# generate data
X = np.random.randn(500, 25)
y = np.random.randn(500,)

group_index = np.zeros(500,)
group_index[150:] = 1

X[:, :10] = X[:, :10] + 2 * y.reshape(-1, 1)

X[:150, 5:20] += 3 * np.random.randn(150, 15).dot(np.random.randn(15, 15))
X[150:, 5:20] += 5 * np.random.randn(350, 15).dot(np.random.randn(15, 15))

# define groupICA-based pipeline
model_groupICA = Pipeline(steps=[
    ('groupICA', GroupICA(n_components=10)),
    ('regression', LinearRegression())])

# get cross-validated predictions with groupICA-based pipeline
y_hat_groupICA = cross_val_predict(
    model_groupICA,
    X,
    y,
    fit_params={'groupICA__group_index': group_index})

# define uwedgeICA-based pipeline (second-order-based, ignores groupstructure)
model_uwedgeICA = Pipeline(steps=[
    ('uwedgeICA', UwedgeICA(n_components=10)),
    ('regression', LinearRegression())])

# get cross-validated predictions with uwedgeICA-based pipeline
y_hat_uwedgeICA = cross_val_predict(
    model_uwedgeICA,
    X,
    y)

# define pooled fastica-based pipeline (ignores groupstructure)
model_fastica = Pipeline(steps=[
    ('fastica', FastICA(n_components=10)),
    ('regression', LinearRegression())])

# get cross-validated predictions with pooled fastica-based pipeline
y_hat_fastica = cross_val_predict(
    model_fastica,
    X,
    y)

# for comparison plot scatter of predictions against the true y
plt.plot(y, y_hat_groupICA,
         '.',
         label='groupICA (correlation with true y {:.2f})'.format(
             np.corrcoef(y, y_hat_groupICA)[0, 1]))

plt.plot(y, y_hat_uwedgeICA,
         '.',
         label='uwedgeICA (correlation with true y {:.2f})'.format(
             np.corrcoef(y, y_hat_uwedgeICA)[0, 1]))

plt.plot(y, y_hat_fastica,
         '.',
         label='pooled fastica (correlation with true y {:.2f})'.format(
             np.corrcoef(y, y_hat_fastica)[0, 1]))

plt.title('y vs y_hat')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('true y')
plt.ylabel('y_hat')

plt.show()
