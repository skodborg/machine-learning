"""
This file is the basis for your implementation of the K-means algorithm.
As a bare minimum, you should replace TODO with your own code,
but feel free to reorganize the code as you see fit.

Last modified November 30 (second release).
"""

from __future__ import unicode_literals, division

import numpy as np


def closest(data, centers):
    """
    Compute the closest center of each data point.

    Parameters
    ----------
    points : (n, d) array
        The input data
    centers : (k, d) array
        The computed centers

    Returns
    -------
    rep : (n,) array with integers from 0 to k - 1
        For each point, the index of the closest center
    """

    # Validate input
    n, d = data.shape
    k, d_ = centers.shape
    assert d == d_

    TODO


def kmeans_cost(data, rep, centers):
    """
    K-means cost measure.

    Parameters
    ----------
    data : (n, d) array
        The input data
    rep : (n,) array containing integers from 0 to k - 1
        The representative of each data point
    centers : (k, d) array
        The representative points

    Returns
    -------
    cost : float
        The cost measure for the given clustering
    """

    TODO


def kmeans(data, k, epsilon):
    """
    K-means algorithm for clustering.

    Parameters
    ----------
    data : (n, d) array
        The input data
    k : int
        The number of clusters
    epsilon : float
        Stop iteration when no center moves by more than epsilon in 1 iteration

    Returns
    -------
    centers : (k, d) array
        The centroids
    """

    data = np.asarray(data)
    n, d = data.shape

    # Initialize centers
    TODO

    tired = False
    while not tired:
        old_centers = centers

        TODO

        dist = np.sqrt(((centers - old_centers) ** 2).sum(axis=1))
        tired = np.max(dist) <= epsilon

    return centers
