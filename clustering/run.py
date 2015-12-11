"""
This file contains an example of how to run K-means and EM
on the 2-dimensional Iris data set.

You can change and extend this file as you see fit in order to run your
clustering experiments.
"""

from __future__ import unicode_literals, division

import numpy as np
from silhouette import silhouette, plot_silhouette
from f1 import f1
from load_and_show_iris import load_iris, load_iris_pca
from em import most_likely, em
from kmeans import kmeans, closest
import matplotlib.pyplot as plt


def evaluate(ax, name, data, predicted, labels):
    F_individual, F_overall, contingency = f1(predicted, labels)
    s = silhouette(data, predicted)

    print("%s F1=%s" % (name, F_overall,))
    plot_silhouette(ax, s, predicted, label=name)


def main():
    data, labels = load_iris_pca()

    k = np.max(labels) + 1
    # fixed_mean = np.array([[-3.59, 0.25],[-1.09,-0.46],[0.75,1.07]])
    mean, cov, prior = em(data, k, 1e-8)
    count = 0
    running = True
    while running:
        fig, ax = plt.subplots()

        centers = kmeans(data, k, 1e-20)
        predicted = closest(data, centers)

        evaluate(ax, 'K-means', data, predicted, labels)

        mean, cov, prior = em(data, k, 1e-8)
        predicted = most_likely(data, mean, cov, prior)

        evaluate(ax, 'EM', data, predicted, labels)

        plt.show()
        count += 1
        if count > 10:
            running = False


if __name__ == "__main__":
    main()
