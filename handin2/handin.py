import os

import numpy as np

import nn


def predict(digits):
    # digits is guaranteed to be an (n x 784) array
    n, nin = digits.shape

    if os.path.exists('weights.npz'):
        # The key 'models' contains ten one-vs-all models
        models = np.load('weights.npz')['models']
    else:
        # Hmm, we have no model. Pick 10 random models
        models = [
            nn.random_neural_net(nin, 40, 1)
            for _ in range(10)
        ]

    probs = []
    for w in models:
        p = nn.nn_predict(w, digits)
        probs.append(p)
    # probs is 10 x n. Pick the largest probability in each
    labels = np.argmax(probs, axis=0)
    return labels


def main():
    pass


if __name__ == "__main__":
    main()
