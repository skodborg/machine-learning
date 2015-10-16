import os

import numpy as np

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.externals import joblib


def predict(digits):
    # digits is guaranteed to be an (n x 784) array

    # load our model
    clf = joblib.load('au_trained_svm.pkl')
    pca = joblib.load('au_trained_pca.pkl')

    # reduce dimensionality of images to 200 features
    digits = pca.transform(digits)

    # predict digits
    predictions = clf.predict(digits)

    return predictions

def main():
    pass


if __name__ == "__main__":
    main()
