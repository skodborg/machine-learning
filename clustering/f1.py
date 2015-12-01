"""
This file is the basis for your implementation of the F1 computation.
You should replace TODO with your own code.
"""

from __future__ import unicode_literals, division

import numpy as np


def f1(predicted, labels):
    """
    Given the predicted labels and the actual labels, compute the F1
    coefficient.

    Parameters
    ----------
    predicted : (n,) array with integers from 0 to r - 1
        The predicted labels
    labels : (n,) array with integers from 0 to k - 1
        The actual labels

    Returns
    -------
    F_individual : (r,) array
        The F1 coefficient of each predicted cluster
    F_overall : float
        The overall F1 coefficient
    contingency : (r, k) array
        The contingency matrix
    """

    n, = predicted.shape
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    TODO
