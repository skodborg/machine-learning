"""
This file is the basis for your implementation of the EM algorithm.
As a bare minimum, you should replace TODO with your own code,
but feel free to reorganize the code as you see fit.
"""

from __future__ import unicode_literals, division

import numpy as np
from numpy.linalg import LinAlgError
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def pdf(points, mean, cov, prior):
    """
    Given the mean, covariance and prior probabilities of the Gaussian mixture
    model, compute the probability densities of each data point.

    Parameters
    ----------
    points : (n, d) array
        The input data, e.g. IRIS data points (d = 2 or 4)
    mean : (k, d) array
        Means of the Gaussians
    cov : (k, d, d) array
        Covariance matrices of the Gaussians
    prior : (k,) array
        Prior probabilities

    Returns
    -------
    prob : (n, k) array
        The probability density of each point and cluster.
        Each row sums to 1.
    """

    # Input handling/validation
    points, mean, cov = np.asarray(points), np.asarray(mean), np.asarray(cov)
    prior = np.asarray(prior)
    n, d = points.shape
    k, d_1 = mean.shape
    k_2, d_2, d_3 = cov.shape
    k_3, = prior.shape
    assert d == d_1 == d_2 == d_3
    assert k == k_2 == k_3, "%s %s %s should be equal" % (k, k_2, k_3)

    # Compute probabilities
    prob = []
    for i in range(k):
        if prior[i] < 1 / k ** 3:
            prob.append(np.zeros(n))
        else:
            prob.append(
                prior[i] *
                multivariate_normal.pdf(
                    mean=mean[i], cov=cov[i], x=points))
    prob = np.transpose(prob)  # n x k
    # Normalize cluster probabilities of each point
    prob = prob / np.sum(prob, axis=1, keepdims=True)  # n x k

    assert prob.shape == (n, k)
    assert np.allclose(prob.sum(axis=1), 1)
    return prob


def most_likely(points, mean, cov, prior):
    """
    Given the mean, covariance and prior probabilities of the Gaussian mixture
    model, compute the most likely class of each data point.

    Parameters
    ----------
    points : (n, d) array
        The input data, e.g. IRIS data points (d = 2 or 4)
    mean : (k, d) array
        Means of the Gaussians
    cov : (k, d, d) array
        Covariance matrices of the Gaussians
    prior : (k,) array
        Prior probabilities

    Returns
    -------
    predicted : (n,) array
        For each point, the most likely class
    """

    prob = pdf(points, mean, cov, prior)
    return np.argmax(prob, axis=1)


def em(points, k, epsilon, mean=None, f=None):
    """
    Expectation-Maximization algorithm for clustering in the Gaussian mixture
    model.

    Parameters
    ----------
    points : (n, d) array
        The input data, e.g. IRIS data points (d = 2 or 4)
    k : int
        The number of classes (3)
    epsilon : float
        Stop iteration when no mean moves by more than epsilon in an iteration
    mean : (k, d) array, optional
        Initial placement of Gaussian means
    f : callable accepting (mean, cov, prior), optional
        If specified, receives the parameters after each EM iteration

    Returns
    -------
    mean : (k, d) array
        The means of the predicted Gaussians
    cov : (k, d, d) array
        The covariance matrices of the predicted Gaussians
    prior : (k,) array
        The prior probability of a point belonging to a cluster
    """

    points = np.asarray(points)
    n, d = points.shape

    # Initialize and validate mean
    if mean is None:
        # Randomly pick k points
        TODO

    # Validate input
    mean = np.asarray(mean)
    k_, d_ = mean.shape
    assert k == k_
    assert d == d_

    # Initialize cov, prior
    TODO

    tired = False
    while not tired:
        old_mean = mean
        if f:
            f(mean, cov, prior)

        # Expectation step
        TODO

        # Maximization step
        TODO

        # Finish condition
        dist = np.sqrt(((mean - old_mean) ** 2).sum(axis=1))
        tired = np.max(dist < epsilon)

    # Validate output
    assert mean.shape == (k, d)
    assert cov.shape == (k, d, d)
    assert prior.shape == (k,)
    return mean, cov, prior
