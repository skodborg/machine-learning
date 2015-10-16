# encoding: utf-8
"""
Sample implementation of neural nets.
The main interface is the two functions:
    nn_train(digits, labels, nhidden, [regularization=0.001])
    nn_predict(w, digits)

Use nn_train to compute w, the neural net weights.
Use nn_predict to perform predictions given weights.

>>> n, d = 1000, 20
>>> points = np.random.uniform(-1, 1, (n, d))
>>> norms = (points ** 2).sum(axis=1)
>>> labels = np.minimum(1, np.maximum(0, norms * 20 - 10))
>>> w = nn_train(points, labels, 40)  # doctest: +ELLIPSIS
Initialize random weights (20 inputs, 40 hidden, 1 output)
...
>>> labelpred = nn_predict(w, points)
>>> np.sum((labelpred - labels) ** 2)  # doctest: +SKIP
7.2815586269205532e-06
"""

import time
import argparse
import contextlib

import numpy as np
import scipy.optimize

import load_data


class Timer(object):
    def __init__(self):
        self.n = {}
        self.t = {}

    @contextlib.contextmanager
    def __call__(self, key):
        t0 = time.time()
        yield
        t1 = time.time()
        self.n.setdefault(key, 0)
        self.n[key] += 1
        self.t.setdefault(key, 0)
        self.t[key] += t1 - t0

    def __str__(self):
        times = {k: self.t[k] / self.n[k]
                 for k in self.t}
        times = sorted(times.items(), key=lambda o: -o[1])
        return ', '.join('%s: %.0f ms' % (k, 1000 * v)
                         for k, v in times
                         if v > 1/2 * times[0][1])


timer = Timer()


def sigmoid(x):
    p = 1 / (1 + np.exp(np.abs(x)))
    n = 1 - p
    return np.choose(x < 0, (n, p))


def random_neural_net(nin, nhidden, nout):
    print("Initialize random weights (%d input%s, %d hidden, %d output%s)" %
          (nin, '' if nin == 1 else 's', nhidden,
           nout, '' if nout == 1 else 's'))
    r = np.sqrt(6) / np.sqrt(nin + nhidden + nout)
    w1 = np.random.uniform(-r, r, (nin, nhidden))
    w2 = np.random.uniform(-r, r, (nhidden, nout))
    b1 = np.random.uniform(-r, r, (1, nhidden))
    b2 = np.random.uniform(-r, r, (1, nout))
    return {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2,
            'lambda': 0.001, 'alpha': 1.0, 'sample': 0}


def feed_forward(w, x):
    """
    x should be stored in order='C', that is, row-major order.
    """
    w1 = w['w1']
    w2 = w['w2']
    b1 = w['b1']
    b2 = w['b2']
    assert x.ndim == 2
    assert x.shape[1] == w1.shape[0]
    assert w1.shape[1] == w2.shape[0]
    nin, nhidden = w1.shape
    nhidden, nout = w2.shape
    with timer('hidden'):
        # (n x nin) @ (nin x nhidden)
        # (60000 x 784) @ (784 x 200)
        # 818065408
        hidden_act = sigmoid(np.dot(x, w1) + b1)
    with timer('prediction'):
        # (n x nhidden) @ (nhidden x nout)
        act = sigmoid(np.dot(hidden_act, w2) + b2)
    return {'hidden': hidden_act, 'prediction': act}


def nn_error(labels, activations):
    prediction = activations['prediction']
    log_pred = np.log(prediction)
    log_npred = np.log(1 - prediction)
    cross_entropy = labels * log_pred + (1 - labels) * log_npred
    return -np.mean(np.sum(cross_entropy, axis=1))


def w_norm_sq(w):
    w1 = w['w1']
    w2 = w['w2']
    return np.sum(w1 ** 2) + np.sum(w2 ** 2)


def nn_cost(w, labels, activations):
    l = w['lambda']
    regularization = l/2 * w_norm_sq(w)
    return nn_error(labels, activations) + regularization


def nn_backprop(w, x, labels, activations):
    """
    x should be stored in order='F', that is, column-major order.
    """
    n = labels.shape[0]
    w1 = w['w1']  # nin x nhidden
    nin, nhidden = w1.shape
    w2 = w['w2']  # nhidden x nout
    nhidden, nout = w2.shape
    # b1 = w['b1']
    # b2 = w['b2']
    hidden = activations['hidden']  # n x nhidden
    prediction = activations['prediction']  # n x nout
    l = w['lambda']

    simple = False
    if simple:
        d3 = (1 - labels) * prediction - labels * (1 - prediction)
        d2 = np.dot(d3, w2.T) * (hidden * (1 - hidden))
        gradw2 = 1/n * np.dot(hidden.T, d3)
        gradw2 += l * w2
        gradb2 = 1/n * np.dot(np.ones((1, n)), d3)
        gradw1 = 1/n * np.dot(x.T, d2)
        gradw1 += l * w1
        gradb1 = 1/n * np.dot(np.ones((1, n)), d2)
    else:
        # Optimization: Split up the matrix product
        step = 150

        # n x nout
        with timer('d3'):
            d3 = (1 - labels) * prediction - labels * (1 - prediction)

        # n x nhidden
        with timer('d2'):
            d2 = np.dot(d3, w2.T) * (hidden * (1 - hidden))
        # nhidden x nout
        with timer('gradw2'):
            gradw2 = np.zeros((nhidden, nout))
            for i1 in range(0, n, step):
                i2 = min(n, i1 + step)
                gradw2 = 1/n * np.dot(hidden.T[:, i1:i2], d3[i1:i2, :])
            gradw2 += l * w2
        # 1 x nout
        with timer('gradb2'):
            gradb2 = 1/n * np.dot(np.ones((1, n)), d3)
        # (nin x n) @ (n x nhidden) = nin x nhidden
        with timer('gradw1'):
            gradw1 = np.zeros((nin, nhidden))
            for i1 in range(0, n, step):
                i2 = min(n, i1 + step)
                gradw1 += 1/n * np.dot(x.T[:, i1:i2], d2[i1:i2, :])
            gradw1 += l * w1
        # 1 x nhidden
        with timer('gradb1'):
            gradb1 = 1/n * np.dot(np.ones((1, n)), d2)

    return {'w1': gradw1, 'w2': gradw2,
            'b1': gradb1, 'b2': gradb2}


def nn_descent(w, x, labels):
    """
    Relatively simple implementation of gradient descent with fixed learning
    rate. This does not use the fast minimization methods of scipy.optimize.
    """
    assert x.ndim == 2
    n = x.shape[0]
    x_C = np.array(x, order='C')
    x_F = np.array(x, order='F')
    alpha = w['alpha']
    l = w['lambda']
    maxiter = 100
    for i in range(maxiter):
        activations = feed_forward(w, x_C)
        if labels.shape[1] == 1:
            c = (activations['prediction'] > 0.5)
            incorrect = c.ravel() != (labels.ravel() > 0.5)
        else:
            c_i = np.argmax(activations['prediction'], axis=1)
            l_i = np.argmax(labels, axis=1)
            incorrect = c_i != l_i
        e = incorrect.sum() / n
        cost = nn_error(labels, activations)
        grad = nn_backprop(w, x_F, labels, activations)
        print("[%3d/%3d] e=%4.1f%% cost=%13.7e |w|²=%13.7e |g|²=%13.7e" %
              (i + 1, maxiter, 100 * e, cost, w_norm_sq(w), w_norm_sq(grad)))
        # print(timer)
        w['w1'] -= alpha * grad['w1'] + l * w['w1']
        w['w2'] -= alpha * grad['w2'] + l * w['w2']
        w['b1'] -= alpha * grad['b1']
        w['b2'] -= alpha * grad['b2']
    print(timer)
    return w


def nn_descent_2(w, x, labels):
    """
    Variant of nn_descent that uses scipy.optimize.
    """
    l = w['lambda']
    x = np.asarray(x)
    x_C = np.array(x, order='C')
    x_F = np.array(x, order='F')
    labels = np.asarray(labels)
    KEYS = 'w1 w2 b1 b2'.split()
    n = x.shape[0]

    def flat(w):
        return np.concatenate([w[k].ravel() for k in KEYS])

    def unflat(w_flat):
        res = dict(w)
        for k in KEYS:
            n = len(res[k].ravel())
            res[k] = w_flat[:n].reshape(res[k].shape)
            w_flat = w_flat[n:]
        return res

    i = [0]

    def f(w_flat):
        w = unflat(w_flat)
        if w['sample']:
            indices = np.random.choice(n, w['sample'], replace=False)
            x_samp_C = np.array(x[indices], order='C')
            x_samp_F = np.array(x[indices], order='F')
            l_samp = labels[indices]
            n_samp = w['sample']
        else:
            x_samp_C = x_C
            x_samp_F = x_F
            l_samp = labels
            n_samp = n
        activations = feed_forward(w, x_samp_C)
        if l_samp.shape[1] == 1:
            c = (activations['prediction'] > 0.5)
            incorrect = c.ravel() != (l_samp.ravel() > 0.5)
        else:
            c_i = np.argmax(activations['prediction'], axis=1)
            l_i = np.argmax(l_samp, axis=1)
            incorrect = c_i != l_i
        e = incorrect.sum() / n_samp
        cost = nn_cost(w, l_samp, activations)
        grad = nn_backprop(w, x_samp_F, l_samp, activations)
        regularization = l/2 * w_norm_sq(w)

        i[0] += 1
        # print("[%3d] e=%4.1f%% cost=%13.7e ½λ|w|²=%13.7e |g|²=%13.7e" %
        #       (i[0], 100 * e, cost - regularization, regularization,
        #        w_norm_sq(grad)))
        # print(timer)

        grad_flat = flat(grad)

        debug_gradient = False
        if debug_gradient:
            eps = np.zeros_like(grad_flat)
            eps[np.random.choice(len(eps), 100)] = 1e-6
            cp = nn_cost(
                unflat(w_flat + eps), labels,
                feed_forward(unflat(w_flat + eps), x))
            cm = nn_cost(
                unflat(w_flat - eps), labels,
                feed_forward(unflat(w_flat - eps), x))

            grad_approx = (cp - cm) / 2  # / np.sqrt(np.sum(eps**2))
            grad_reported = np.dot(grad_flat, eps)  # / np.sqrt(np.sum(eps**2))
            print('%s/%s=%s' %
                  (grad_reported, grad_approx, grad_reported / grad_approx))

        return cost, grad_flat

    res = scipy.optimize.minimize(
        f, flat(w), jac=True, method='L-BFGS-B')
    f(res.x)
    return unflat(res.x)


def nn_train(digits, labels, nhidden, regularization=0.001):
    digits = np.asarray(digits)
    labels = np.asarray(labels)

    if digits.ndim > 2:
        raise TypeError("digits has shape %r; should be 2-dimensional" %
                        (digits.shape,))
    elif digits.ndim < 2:
        raise TypeError(
            "digits has not enough dimensions; should be 2-dimensional")

    if labels.ndim < 2:
        labels = labels.reshape((-1, 1))
    elif labels.ndim > 2:
        raise TypeError("labels has shape %r; should be 2-dimensional" %
                        (labels.shape,))

    if digits.shape[0] != labels.shape[0]:
        raise TypeError(
            "Number of digits (%d) does not match number of labels (%d)" %
            (digits.shape[0], labels.shape[0]))
    n, nin = digits.shape
    n, nout = labels.shape
    w = random_neural_net(nin, nhidden, nout)
    w['lambda'] = regularization
    return nn_descent_2(w, digits, labels)


def nn_predict(w, digits):
    activations = feed_forward(w, digits)
    return activations['prediction']


def simple_2_vs_7(digits, labels):
    labels = labels.ravel()
    l2 = (labels == 2)
    l7 = (labels == 7)
    x = np.array(digits[l2 | l7], order='F')
    print("Training 2 vs. 7 classifier on %d points, d=%d" % x.shape)
    y = (labels[l2 | l7] == 2).astype(np.float64)
    y = y.reshape((-1, 1))

    n, nin = digits.shape
    nhidden = 200
    nout = 1
    w = random_neural_net(nin, nhidden, nout)
    w = nn_descent_2(w, x, y)
    return w


def simple_2_vs_all(digits, labels):
    labels = labels.ravel()
    l2 = (labels == 2)
    l7 = (labels != 2)
    x = np.array(digits[l2 | l7], order='F')
    print("Training 2 vs. all classifier on %d points, d=%d" % x.shape)
    y = (labels[l2 | l7] == 2).astype(np.float64)
    y = y.reshape((-1, 1))

    n, nin = digits.shape
    nhidden = 200
    nout = 1
    w = random_neural_net(nin, nhidden, nout)
    w = nn_descent_2(w, x, y)
    return w


def classify_all(digits, labels):
    print("Training 10-digit classifier on %d points, d=%d" % digits.shape)
    labels = labels.reshape((-1, 1))
    x = digits
    y = (labels == np.arange(10).reshape((1, 10))).astype(np.float64)

    n, nin = digits.shape
    nhidden = 20
    nout = 10
    w = random_neural_net(nin, nhidden, nout)
    w = nn_descent_2(w, x, y)
    return w


def autoencoder(digits, labels):
    print("Training autoencoder on %d points, d=%d" % digits.shape)
    n, nin = digits.shape
    nhidden = 200
    nout = nin
    w = random_neural_net(nin, nhidden, nout)
    w = nn_descent_2(w, digits, digits)
    return w


def test(digits, labels):
    print("TEST")
    w = random_neural_net(1, 1, 2)
    w = nn_descent_2(w, [[0], [1]], [[1, 0], [0, 1]])

    print("TEST")
    n = 100
    d1 = 40
    d2 = 10
    d3 = 40
    w = random_neural_net(d1, d2, d3)
    x = np.random.random((n, d1))
    x -= np.mean(x, axis=0, keepdims=True)
    y = (np.arange(d3).reshape((1, d3)) ==
         (np.argmax(x, axis=1) % d3).reshape((n, 1)))
    y = 1.0 * y
    w = nn_descent_2(w, x, y)
    return w

def run_learning():
    t_imgs, t_lbls, v_imgs, v_lbls = load_data.auDigit_data()

    # reshape labels to dimension N x k with k = 10 
    #   (k = 10 because digits 0-9 should each be represented by 
    #    an entry with a value between 0-1, like softmax)
    old_t_lbls = t_lbls   # keep original t_lbls to compare with predictions
    t_lbls = t_lbls.reshape((-1, 1))
    t_lbls = (t_lbls == np.arange(10).reshape((1, 10))).astype(np.float64)

    for h in [25, 40, 50, 75, 100, 150, 200, 500]:
        for i in [-15, -14, -13, -12, -11, -10, -9, -8]:
            print("\nTrying hidden = " + str(h) + " and reg = 3**" + str(i))

            # train neural net
            nhidden = h
            reg = 3**i
            w = nn_train(t_imgs, t_lbls, nhidden, regularization=reg)

            # estimate in-sample performance
            predictions = nn_predict(w, t_imgs)
            predictions = np.argmax(predictions,axis=1).astype(np.float64)    
            tests = old_t_lbls.shape[0]
            hits = np.sum(predictions == old_t_lbls)
            pct = hits / tests * 100
            print('in-sample:\n' + str(hits) + '/' + str(tests) + ', ' + str(pct) + ' %')

            # estimate out-of-sample performance by validation of prediction
            predictions = nn_predict(w, v_imgs)
            predictions = np.argmax(predictions,axis=1).astype(np.float64)    
            tests = v_lbls.shape[0]
            hits = np.sum(predictions == v_lbls)
            pct = hits / tests * 100
            print('out-of-sample:\n' + str(hits) + '/' + str(tests) + ', ' + str(pct) + ' %')
    

def main():
    MODES = {'2vs7': simple_2_vs_7, 'all': classify_all,
             '2vsall': simple_2_vs_all,
             'autoenc': autoencoder, 'test': test}
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--mode', choices=MODES, default='2vs7')
    args = parser.parse_args()

    data = dict(np.load(args.filename))
    if 'digits' in data:
        digits = data.pop('digits')
    elif 'images' in data:
        digits = data.pop('images')
    else:
        print(data.keys())
        raise KeyError('digits')
    labels = data.pop('labels')
    if data:
        print("Unknown data keys %r" % (data.keys(),))

    MODES[args.mode](digits, labels)


if __name__ == "__main__":
    # main()
    run_learning()
