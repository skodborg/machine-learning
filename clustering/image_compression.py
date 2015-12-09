"""
This file contains an example of how to run kmeans on the DAIMI image.

You can extend this file as you see fit in order to solve the
image compression exercise.
"""

from __future__ import unicode_literals, division

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from kmeans import kmeans, closest


def daimi_kmeans(k=4):
    im = scipy.ndimage.imread('AU_main_back_small.png') / 255
    height, width, depth = im.shape

    data = im.reshape((height * width, depth))
    centers = kmeans(data, k, 1e-2)
    rep = closest(data, centers)
    data_compressed = centers[rep]

    im_compressed = data_compressed.reshape((height, width, depth))
    plt.figure()
    plt.imshow(im_compressed)
    plt.show()


if __name__ == "__main__":
    daimi_kmeans()
