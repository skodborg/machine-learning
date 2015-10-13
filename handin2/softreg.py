import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.special as scisp
import scipy.optimize as opti


def softmax(X):
    mx = np.amax(X,axis=1,keepdims=True)
    tmp = np.log(np.sum(np.exp(X - mx),axis=1,keepdims=True)) + mx
    return np.exp(X - tmp)
    
    
def soft_cost(X,Y,Wl,reg=0):
    W = Wl.reshape(X.shape[1],-1)
    pred = softmax(np.dot(X,W)) # sigma(XW) all predictions
    # reg cost
    reg_cost = 0.5*reg*np.sum(W[1:,:]**2)
    # Cost (NLL) three different eqiuvalent versions
    cost = -np.sum(Y * np.log(pred))
    #cost = -np.sum(np.log(np.sum(pred*Y,axis=1)))
    # _,yidx = Y.nonzero()
    # cost = -np.sum(np.log(pred[np.arange(Y.shape[0]),yidx]))+reg_cost
    # Gradient
    err = Y-pred # difference between data and prediction
    grad = -np.dot(X.T,err)
    reg_grad = reg * W
    reg_grad[0,:] = 0 #First row is the bias variables and are not charged in reg cost
    print('cost, reg_cost ',cost,reg_cost)
    # normalize cost and gradient to data size
    return reg_cost + cost/X.shape[0],reg_grad.reshape(-1,) + grad.reshape(-1,)/X.shape[0]

def grad_descent(X,y,w=None,reg=0.0,rounds =10):
    if w is None: w = np.zeros((X.shape[1],y.shape[1])).reshape(-1,)
    lr = 1
    for i in range(rounds):
        cost,grad = soft_cost(X,y,w,reg)
        print('Iteration {0}: Average Cost {1}'.format(i,cost/y.size))
        #print('norm of gradient {0}'.format(np.linalg.norm(grad)))
        w = w - lr * grad
    return w

def fast_descent(X,y,w=None,reg=0,rounds=5):
    # unstable if linear separable...
    if w is None: w = np.zeros((X.shape[1],y.shape[1])).reshape(-1,)
    print('fast descent',w.shape)
    w =  opti.minimize(lambda t: soft_cost(X,y,t,reg),w,jac=True,method='L-BFGS-B',options={'maxiter':rounds,'disp':True})
    return w.x

def test_descent():
    x0 = np.ones(2)
    w =  opti.minimize(lambda t: np.sum(t**2),x0,jac=False)
    
def test():
    X = np.ones((3,3))
    X[0,0] = 2
    lg = sc.misc.logsumexp(X)
    print('log sum ',lg)
    sm = softmax(X)
    print('softmax of 1,2,3',sm)

if __name__ == "__main__":
    test()
