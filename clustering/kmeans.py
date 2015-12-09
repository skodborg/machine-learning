"""
This file is the basis for your implementation of the K-means algorithm.
As a bare minimum, you should replace TODO with your own code,
but feel free to reorganize the code as you see fit.

Last modified November 30 (second release).
"""

from __future__ import unicode_literals, division

import numpy as np
from load_and_show_iris import load_iris, load_iris_pca

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

  result = np.zeros(n).astype('int')
  for i, p in enumerate(data):
    curr_min_dist = np.float('inf')
    corresponding_idx = 0
    for j, c in enumerate(centers):
      distance = np.sum((p - c)**2)
      if distance < curr_min_dist:
        curr_min_dist = distance
        corresponding_idx = j
    result[i] = corresponding_idx
  assert result.shape[0] == n
  return result



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

  n, d = data.shape
  n_ = rep.shape[0]
  k, d_ = centers.shape
  assert d == d_
  assert n == n_

  cost = 0.0
  for i, p in enumerate(data):
    c = centers[rep[i]]
    distance = np.sum((p - c)**2)
    cost += distance

  return cost
  


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

  # Initialize centers to k random points from the data set
  centers = data[np.random.choice(np.arange(0, len(data)), k, replace=False)]

  tired = False
  while not tired:
    old_centers = np.copy(centers)

    # assign x_j to closest centroid
    rep = closest(data, centers)

    # update centroids
    for i, p in enumerate(data):
      centers[rep[i]] += p

    rep_list = list(rep)
    for i in range(0, k):
      centers[i] = centers[i] / rep_list.count(i)

    # printing cost to verify implementation (should descend as we iterate)
    # print(kmeans_cost(data, rep, centers))
    
    # until
    dist = np.sqrt(((centers - old_centers) ** 2).sum(axis=1))
    tired = np.max(dist) <= epsilon

  return centers
