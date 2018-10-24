[![PyPI version](https://badge.fury.io/py/groupICA.svg)](https://badge.fury.io/py/groupICA)
[![Build Status](https://travis-ci.org/sweichwald/groupICA-python.svg?branch=master)](https://travis-ci.org/sweichwald/groupICA-python)

# groupICA-python

Please refer to the project website at https://sweichwald.de/groupICA/.
We kindly ask you to cite the accompanying article (see below), in case this package should prove useful for some work you are publishing.

Quick install

    pip install groupICA

The developer documentation is available at https://sweichwald.de/groupICA-python.

This repository holds the source of the [groupICA package](https://pypi.org/project/groupICA/) which implements the groupICA algorithm presented in
[groupICA: Independent component analysis for grouped data](https://arxiv.org/abs/1806.01094) by N Pfister*, S Weichwald*, P Bühlmann, B Schölkopf.

Furthermore, as a courtesy to other python users, this package contains implementations of
* uwedge, an approximate matrix joint diagonalisation algorithm described [here](https://doi.org/10.1109/TSP.2008.2009271), and
* `uwedgeICA`, which essentially—for the right choice of `timelag` parameters—amounts to an implementation of several second-order-statistics-based ICA algorithms such as SOBI/NSS-JD/NSS-TD-JD (please refer to the groupICA article mentioned above for more details on this),
which may be helpful in their own right independent of the grouped ICA.