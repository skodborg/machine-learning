"""
This file is the basis for your implementation of the F1 computation.
You should replace TODO with your own code.
"""

from __future__ import unicode_literals, division

import numpy as np

def euclid_dist(p, q):
    dim = p.shape[0]
    sum_sqrd_diff = 0.0
    for i in range(0,dim):
        sum_sqrd_diff += (p[i] - q[i])**2
    return np.sqrt(sum_sqrd_diff)

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

    mu_in = []
    mu_out = []

    # mean distance from point p to points q in its own cluster
    for i, p in enumerate(data):
        distance = 0.0
        p_cluster = predicted[i]
        for j, q in enumerate(data):
            q_cluster = predicted[j]
            if p_cluster == q_cluster:
                distance += euclid_dist(p, q)
        mean = distance / (list(predicted).count(p_cluster) - 1)
        mu_in.append(mean)

    # mean distance from point p to points q in closest cluster
    for i, p in enumerate(data):
        distances = {}
        cluster_sizes = {}
        p_cluster = predicted[i]
        for j, q in enumerate(data):
            q_cluster = predicted[j]
            if p_cluster == q_cluster:
                continue
            else:
                if q_cluster not in distances:
                    distances[q_cluster] = 0.0
                    cluster_sizes[q_cluster] = 0
                distances[q_cluster] += euclid_dist(p, q)
                cluster_sizes[q_cluster] += 1
        for k, v in distances.items():
            distances[k] = v / cluster_sizes[k]
        
        mu_out.append(distances[min(distances, key=distances.get)])

    # silhouette coefficient, s_i, for each point i
    silhouette_coeff = []
    for i, v_in in enumerate(mu_in):
        v_out = mu_out[i]
        max_v = max(v_in, v_out)
        s_i = (v_out - v_in)/max_v
        silhouette_coeff.append(s_i)
    
    s = np.array(silhouette_coeff)

    assert s.shape == (n,)
    return s
