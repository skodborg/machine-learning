"""
This file is the basis for your implementation of the F1 computation.
You should replace TODO with your own code.
"""

from __future__ import unicode_literals, division

import numpy as np


def plot_silhouette(ax, s, predicted, **plot_kwargs):
    indices = np.lexsort((-s, predicted))
    return ax.plot(s[indices], **plot_kwargs)


def silhouette(data, predicted):
    """
    Given the data and the predicted labels, compute the silhouette
    coefficient.

    Parameters
    ----------
    data : (n, d) array
        The data that has been clustered
    predicted : (n,) array with integers from 0 to r - 1
        The predicted labels

    Returns
    -------
    s : (n,) array
        The silhouette coefficient of each data point
    """
    data = np.asarray(data)
    n, d = data.shape
    predicted = np.squeeze(np.asarray(predicted))
    k = np.max(predicted) + 1
    assert predicted.shape == (n,)

    TODO
    s = TODO

    assert s.shape == (n,)
    return s
