"""
This file is the basis for your implementation of the F1 computation.
You should replace TODO with your own code.
"""

from __future__ import unicode_literals, division

import numpy as np

# faked test-data from book p. 428
# fake_labels = np.append(np.append(np.append(np.append(np.ones(47), (np.ones(14)*2)), np.zeros(50)), np.ones(3)), (np.ones(36)*2)).astype('int')
# fake_predic = np.append(np.append(np.append(np.append(np.zeros(47), np.zeros(14)), np.ones(50)), (np.ones(3)*2)), (np.ones(36)*2)).astype('int')


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

    # create and fill contingency table
    contingency = np.zeros(r*k).reshape(r,k)
    for i in range(0,n):
        pred = predicted[i]
        lbl = labels[i]
        contingency[pred][lbl] += 1
    
    # calculate F-values
    F_individual = []
    for i in range(0,r):
        n_i = list(predicted).count(i)
        j_i = np.argmax(contingency[i])
        n_iji = contingency[i][j_i]
        m_ji = list(labels).count(j_i)
        F_individual.append(2 * n_iji / (n_i + m_ji))
        
    F_overall = np.sum(F_individual)/r

    return F_individual, F_overall, contingency
