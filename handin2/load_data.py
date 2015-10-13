import numpy as np

def mnist_train():
    data = np.load('mnistTrainT.npz')
    imgs = data['digits']
    imgs = np.c_[np.ones(imgs.shape[0]), imgs]
    lbls = np.squeeze(data['labels'])
    return imgs, lbls

def mnist_test():
    data = np.load('mnistTestT.npz')
    imgs = data['digits']
    imgs = np.c_[np.ones(imgs.shape[0]), imgs]
    lbls = np.squeeze(data['labels'])
    return imgs, lbls