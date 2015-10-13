import numpy as np
import matplotlib.pyplot as plt
import load_data
import softreg as sm

def run(reg):
    # load data and set parameters
    # images,labels =load_data.mnist_train()
    # test_images,test_labels = load_data.mnist_test()

    images, labels, test_images, test_labels = load_data.auDigit_data()
    labels = labels.astype(int)
    test_labels = test_labels.astype(int)

    images = np.c_[np.ones(images.shape[0]), images]
    test_images = np.c_[np.ones(test_images.shape[0]), test_images]

    # reg = 0.0
    rounds = 200
    # shape labels into matrix
    lab_matrix = np.zeros((labels.size,10))
    lab_matrix[np.arange(labels.shape[0]),labels]=1
    # make initial vector
    start_theta = np.zeros((images.shape[1],lab_matrix.shape[1])).reshape(-1,)
    # run learning algorithm
    opt_theta = sm.fast_descent(images,lab_matrix,start_theta,reg,rounds)

    # Compute in sample accuracy
    pred = np.dot(images,opt_theta.reshape(images.shape[1],-1))
    max_predict = np.argmax(pred,axis=1)
    print(np.c_[max_predict,labels])
    print('mean predict sanity test',max_predict.mean())
    acc = (max_predict==labels).mean()
    print('Softmax in sample classification rate {0}'.format(acc*100))

    # Compute test sample accuracy
    test_lab_matrix = np.zeros((test_labels.size,10))
    test_lab_matrix[:,test_labels] = 1
    test_pred = np.dot(test_images,opt_theta.reshape(test_images.shape[1],-1))
    test_max_predict = np.argmax(test_pred,axis=1)
    test_acc = (test_max_predict==test_labels).mean()
    print('Softmax test sample classification rate {0}'.format(test_acc*100))
    # print learned weights
    tr = opt_theta.reshape(785,10)
    tr = tr[1:,:]
    # plt.imshow(tr.reshape(28,-1,order='F'),cmap='bone')
    # plt.show()
    ## Should Print something like
    ## Softmax in sample classification rate 88.82
    ## Softmax test sample classification rate 89.58

if __name__=="__main__":
    for i in [-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0]:
        print("REGULARIZATION: " + str(3) + "**" + str(i))
        run(3**i)
    print("REGULARIZATION: " + str(0))
    run(0)
