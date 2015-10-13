import numpy as np
import matplotlib.pyplot as plt
import logreg
import load_data 
import scipy.optimize as opti

# Not working code that should give some hints

def eval_model(X,y,w):
    """
 Evaluate logistic regression model w on data X and true output y
    """
    pred = np.dot(X,w) > 0
    # Print first 10 predictions and actual value, np.c_ concatenates vectors into a matrix column wise
    print(np.c_[pred,y][0:10,:])
    er = (pred== y) # er is 0,1 vector 1 means true prediction 0 means false.
    # numpy has sum,mean,min,max etc.
    acc = er.mean()
    return acc
    
def run_pair(img,labels,id1,id2,reg=0,rounds=50):
    """
    Runs logistic regression on two digits and plot the vector
    """

    # select the data
    idx1 = (labels == id1)
    idx2 = (labels == id2)
    img27 = img[idx1 | idx2,:]
    lab27 = labels[idx1 | idx2]
    # 1.0 makes it into float array (still of ones and zeros)
    lab27 = (lab27 == id1)*1.0
    
    start_theta = np.zeros(img27.shape[1])
    # assuming logreg exists
    opt_theta = logreg.fast_descent(img27,lab27,start_theta,reg,rounds)
    print('In Sample Classification Rate {0} %'.format(100*eval_model(img27,lab27,opt_theta)))
    # visualize weights (ignoring bias)
    plt.imshow(opt_theta[1:].reshape(28,28),cmap='bone')
    plt.show()


def soft_cost(X,y,Wl,reg):
    # reshape parameters to a N x K matrix
    W = Wl.reshape(X.shape[1],-1)
    # code computing cost, reguarlization cost, gradient and regularized gradient cost
    # return average_cost + reg_cost, reshaped back again gradient + regularization gradient
    return reg_cost + cost/X.shape[0],reg_grad.reshape(-1,) + grad.reshape(-1,)/X.shape[0]

def fast_descent(X,y,w=None,reg=0,rounds=500):
    """ Apply minimize function for softmax
    Softcost return (average) cost and gradient (cost is a number and gradient is a 1d vector shape = (d,)
    Minimize expects 1d Input, so we give it one.
    """    
    if w is None: w = np.zeros((X.shape[1],y.shape[1])).reshape(-1,)    
    w =  opti.minimize(lambda t: soft_cost(X,y,t,reg),w,jac=True,method='L-BFGS-B',options={'maxiter':rounds,'disp':True})
    return w.x # w.x is a vector not a matrix. Rehape It Back if you want matrix

def softmax(X):
    # compute the max of each row in N by d matrix, return N,1 shaped matrix (because of keepdimes)
    mx = np.amax(X,axis=1,keepdims=True)    
    # use to compute softmax  X-mx is matrix of size N x d where max of each row is subtracted
    # secret...
    raise NotImplementedError

if __name__=="__main__":
    img,labels = load_data.au_train()
    # show first 10 digits in one plot
    tmp = img[0:10,1:]
    plt.imshow(tmp.reshape(-1,28),cmap='bone')
    plt.figure()
    tmp2 = np.array([x.reshape(28,28).T.reshape(-1,) for x in tmp])
    plt.imshow(tmp2.reshape(-1,28).T,cmap='bone')
    tmp3 = img[0:10*12, 1:].reshape((10, 12, 28, 28))
    tmp3 = np.rollaxis(tmp3, 2, 1).reshape((10*28, 12*28))
    plt.figure()
    plt.imshow(tmp3, cmap='bone')
    plt.show()
    print('actual labels',labels[0:10])
    #plt.show()
    run_pair(img,labels,2,7,50,0.0001)
