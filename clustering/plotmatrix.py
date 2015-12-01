"""
This file contains two functions for plotting the Iris data set.

Last modified November 25 (second release).
"""

from __future__ import unicode_literals, division

import numpy as np
import matplotlib.pyplot as plt

try:
    # matplotlib 1.3 and earlier
    from matplotlib.axes import _process_plot_format
except ImportError:
    # matplotlib 1.4 and 1.5
    from matplotlib.axes._base import _process_plot_format


def plot_matrix(x, y, group, fmt='.'):
    """
    Given two d-dimensional datasets of n points,
    makes a figure containing d x d plots, where the (i, j) plot
    plots the ith dimension against the jth dimension.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    group = np.squeeze(np.asarray(group))
    n, p = x.shape
    n_, q = y.shape
    n__, = group.shape
    assert n == n_ == n__
    groups = sorted(set(group))
    if isinstance(fmt, str):
        # We need this to differentiate fmt='os^' (len(groups) markers)
        # from fmt='.' (a single marker)
        try:
            _process_plot_format(fmt)
        except ValueError:
            # It is not a valid format, so it is probably
            # an iterable of formats
            pass
        else:
            fmt = {k: fmt for k in groups}
    fig, axes = plt.subplots(p, q, squeeze=False)
    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            for g in groups:
                ax.plot(x[group == g, i], y[group == g, j], fmt[g])
    return fig


def plot_groups(x, group, fmt='.'):
    """
    Helper function for plotting a 2-dimensional dataset with groups
    using plot_matrix.
    """
    n, d = x.shape
    assert d == 2
    x1 = x[:, 0].reshape(n, 1)
    x2 = x[:, 1].reshape(n, 1)
    return plot_matrix(x1, x2, group, fmt)
