import argparse
import numpy as np
import matplotlib.pyplot as plt


def pla_train(X, y, theta=None):
    """
    The perceptron learning algorithm. Should be self explanatory.
    """    
    if theta is None:
        theta = np.zeros(X.shape[1])
    while True:
        wrong = np.sign(np.dot(X, theta)) != y
        if not wrong.any():
            return theta
        print('mispredictions: {0}'.format(np.sum(wrong)))
        indices = wrong.nonzero()[0]
        idx = np.random.choice(indices)
        theta = theta + (y[idx] * X[idx, :])


def make_features(dat, dim):
    """
    Short code for generating polynomial features for 2D data.
    Three nested loops in one list comprehension
    >>> make_features([[2, 3], [0, 0]], dim=4)
    array([[ 1,  3,  9, 27,  2,  6, 18,  4, 12,  8],
           [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    """
    pdat = np.array([[a**i * b**j
                      for i in range(dim)
                      for j in range(dim - i)]
                     for a, b in dat])
    return pdat


def pla_poly_fit(dat, target, dim):
    """
    Code for generating poly features and applying perceptron
    """
    feat = make_features(dat, dim)
    w = pla_train(feat, target)
    print('hyperplane learned is:\n%s' % (w,))
    return w


def linreg_poly_fit(dat, target, dim):
    """
    Code for generating poly features and applying linear regression for classification
    """
    feat = make_features(dat, dim)
    w = np.dot(np.linalg.pinv(feat), target)
    pred = np.sign(np.dot(feat, w))
    wrong = (pred != target)
    targetp = target == +1
    targetm = target == -1
    wrongp = targetp & wrong
    wrongm = targetm & wrong
    print("Linear regression +1 errors: %d/%d" %
          (wrongp.sum(), targetp.sum()))
    print("Linear regression -1 errors: %d/%d" %
          (wrongm.sum(), targetm.sum()))
    return w


def plot_hypothesis(dat, target, w, dim):
    """
    Code for plotting decision boundary in original space
    Generate grid of points, reshape into matrix of x1,x2, add features, apply learned model
    Reshape back to grid and plot contour lines
    """
    sampling = 100
    x = np.linspace(dat[:, 0].min(), dat[:, 0].max(), sampling)
    y = np.linspace(dat[:, 1].min(), dat[:, 1].max(), sampling)
    xx, yy = np.meshgrid(x, y)
    dm = np.c_[xx.reshape(-1), yy.reshape(-1)]
    dfeat = make_features(dm, dim=dim)
    pred = np.dot(dfeat, w)
    pm = pred.reshape(sampling, sampling)

    plt.figure()
    plt.contourf(xx, yy, pm, 50, cmap=plt.get_cmap('bone'))
    plt.scatter(dat[:, 0], dat[:, 1], c=target, cmap=plt.cm.Paired, s=80)
    # lims = plt.axis()
    lims = [dat[:, 0].min(), dat[:, 0].max(), dat[:, 1].min(), dat[:, 1].max()]
    plt.contour(xx, yy, pm, [0], linewidths=3, colors='green')
    plt.axis(lims)


def load_and_plot_data():
    """
    Simple main method that loads data plots it
    and then applies our learningm models.
    Note that this function takes a filename as command line argument
    and you need to close figure to move on from show commands
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    dat = np.load(args.filename)
    dsets = [(dat['x'], dat['y'])]
    for X, y in dsets:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=80)
        lims = [X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()]
        plt.axis(lims)
    plt.show()

    dim = 4
    for X, y in dsets:
        w = pla_poly_fit(X, y, dim)
        plot_hypothesis(X, y, w, dim)
        w = linreg_poly_fit(X, y, dim)
        plot_hypothesis(X, y, w, dim)
    plt.show()


if __name__ == "__main__":
    load_and_plot_data()
