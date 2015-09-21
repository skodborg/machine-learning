import numpy as np
import matplotlib.pyplot as plt
import math
import time

autraindatafile = np.load('auTrainMerged.npz')

autrainlabels = autraindatafile['labels']
autrainimages = autraindatafile['digits']

def softmax(x):
	max_x = np.amax(x)
	expsum = np.sum(np.exp(x - max_x))
	return np.exp(x - (max_x + np.log(expsum)))

def matrix_softmax(X):
	result = []
	for x in X:
		result.append(softmax(x))
	return np.array(result)

# takes a list of digit labels, returns a matrix of 
# size (n, K) with 1's on indexes corresponding to labels in input y
def convert_labels(y):
	classifiers = 10
	# create zero-filled Y
	Y = np.zeros(y.shape[0]*classifiers).reshape(y.shape[0],classifiers)
	# fill in indexes corresponding to labels
	for i in range(0,y.shape[0]):
		Y[i,y[i]] = 1
	return Y

def append_ones_column_first(X):
	return np.c_[np.ones(X.shape[0]), X]

def soft_cost(X, Y, theta):
	fn_cost = lambda X, Y, theta: (1.0 / X.shape[0]) * -(np.sum(np.dot(Y, np.log(matrix_softmax(np.dot(X, theta))).T)))
	gradient = (1.0 / X.shape[0]) * np.dot(-X.T, (Y - matrix_softmax(np.dot(X, theta))))
	return fn_cost, gradient

def soft_run(X, Y, theta, useRegularization=False, lambda_i=0):
	w = theta
	v = np.zeros(1)
	my = 0.000000000005
	costImprovLimit = 0.01
	current_cost = 1000000.0
	costImprovement = 1.0
	t = 0
	iters = []
	costs = []
	# interactive plotting as we progress
	plt.ion()
	plt.show()
	for t in range(0, 20):
	# while costImprovement > costImprovLimit:
		costfn, gradient = soft_cost(X, Y, w,)
		old_cost = current_cost
		current_cost = costfn(X, Y, w)
		# plot cost per iteration
		costs.append(current_cost)
		iters.append(t)

		costImprovement = ((old_cost - current_cost) / old_cost) * 100.0
		print str(t) + ": " + str(current_cost) + "  diff: " + str(costImprovement)
		v = -(gradient)
		w = w + my * v

		# interactive plotting as we progress
		plt.plot(iters, costs, 'r-')
		plt.draw()
		t += 1
	print "Finished, returning gradient at a cost improvement of " + format(costImprovement, '.17f')
	# plot costs per iteration
	plt.plot(iters, costs, 'r-')
	# plt.show()
	return w


someX = np.concatenate([singleX, singleX, singleX]).reshape(3,9)
someY = np.array([0,1,0,1,1,0]).reshape(3,2)
someTheta = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]).reshape(9,2)

# print "X: " + str(someX.shape)
# print "Y: " + str(someY.shape)
# print "Theta: " + str(someTheta.shape)

# print np.dot(someY[3], np.log(softmax(np.dot(someTheta.T, someX[3]))))
# print np.dot(-someX.T, someY - matrix_softmax(np.dot(someX, someTheta)))

myXs = append_ones_column_first(autrainimages)
myYs = convert_labels(autrainlabels)
myTheta = np.zeros(myXs.shape[1]*myYs.shape[1]).reshape(myXs.shape[1], myYs.shape[1])
# soft_run(myXs, myYs, myTheta)

