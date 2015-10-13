import numpy as np

def mnist_train(addBias=True):
    data = np.load('mnistTrainT.npz')
    imgs = data['digits']
    if addBias:
        imgs = np.c_[np.ones(imgs.shape[0]), imgs]
    lbls = np.squeeze(data['labels'])
    return imgs, lbls

def mnist_test(addBias=True):
    data = np.load('mnistTestT.npz')
    imgs = data['digits']
    if addBias:
        imgs = np.c_[np.ones(imgs.shape[0]), imgs]
    lbls = np.squeeze(data['labels'])
    return imgs, lbls

def auDigit_data(ratio=5):
    data = np.load('auTrainMerged.npz')
    imgs = data['digits']
    lbls = data['labels']

    # combine data and shuffle to keep img-lbl pairing
    comb = np.c_[imgs, lbls]
    np.random.shuffle(comb)

    # split into training and validation sets based on ratio
    validation_size = imgs.shape[0]/ratio
    training_size = imgs.shape[0] - validation_size
    training_set = comb[0:training_size,:]
    validation_set = comb[training_size:imgs.shape[0],:]

    # remove labels-column from combined data sets, save as imgs sets
    training_imgs = training_set[:,0:training_set.shape[1]-1]
    validation_imgs = validation_set[:,0:validation_set.shape[1]-1]

    # filter out last column containing labels, save as lbls sets
    training_labels = training_set[:,training_set.shape[1]-1]
    validation_labels = validation_set[:,validation_set.shape[1]-1]
    return training_imgs, training_labels, validation_imgs, validation_labels