"""
This file contains two functions to load the Iris data set
and optionally to apply Principal Components Analysis.
"""

from __future__ import unicode_literals, division

import sklearn.datasets
import sklearn.decomposition
import matplotlib.pyplot as plt

from plotmatrix import plot_matrix, plot_groups


def load_iris():
    iris = sklearn.datasets.load_iris()
    data = iris['data']
    labels = iris['target']
    assert data.shape == (150, 4)
    assert labels.shape == (150,)
    return data, labels


def load_iris_pca():
    data, labels = load_iris()
    pca = sklearn.decomposition.PCA(2)
    return pca.fit_transform(data), labels


def main_pca():
    datapca, labels = load_iris_pca()
    plot_groups(datapca, labels, 'os^')


def main_full():
    data, labels = load_iris()
    plot_matrix(data, data, labels, 'os^')


if __name__ == "__main__":
    main_pca()
    main_full()
    plt.show()
