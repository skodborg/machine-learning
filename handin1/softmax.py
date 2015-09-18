import numpy as np
import matplotlib.pyplot as plt
import math
import time

testdatafile = np.load('mnistTest.npz')

testlabels = np.squeeze(testdatafile['labels'])
testimages = testdatafile['images']


datafile = np.load('mnistTrain.npz')

labels = np.squeeze(datafile['labels'])
images = datafile['images']


trainingimgs = images[0:3,]
traininglabels = labels[0:3,]

def softmax(x):
		# x is a K-dimensional vector, K = 10
		max_x = np.amax(x)
		expsum = np.sum(np.exp(x - max_x))
		return np.array([np.exp(xi - (max_x + np.log(expsum))) for xi in x])

def soft_cost(X, Y, theta):
	# K is classifiers 0-9, i.e. 10
	# X is (n,d+1)
	# y is (n,K)
	# theta is (d+1,K)

	def fn_cost(X, Y, theta):
		# lol = -np.sum(np.dot(Y, np.log(softmax(np.dot(X, theta).T))))
		print X.shape
		print Y.shape
		lol = np.log(softmax(np.dot(X, theta).T)).T
		print np.dot(lol, Y.T)
		# print softmax((np.dot(X, theta))).shape
		# print softmax((np.dot(X, theta).T)).shape
		# print Y.shape


		costsum = 0.0
		for i in range(0, X.shape[0]):
			costsum += np.dot(Y[i], np.log(softmax(np.dot(theta.T, X[i]))))

		print str(-costsum)

		print "difference: " + str(lol+costsum)
		return -costsum

	gradient = -np.dot(X.T, Y - softmax(np.dot(X, theta)))
	
	return fn_cost, gradient


def soft_run(X, Y, theta):
	w = theta
	v = np.zeros(1)
	my = 0.1
	current_cost = 10000000000000000.0
	costImprovLimit = 0.01
	costImprovement = 1.0
	t = 0
	# for t in range(0, 10000):
	while costImprovement > costImprovLimit:
		costfn, gradient = soft_cost(X, Y, w)
		old_cost = current_cost
		current_cost = costfn(X, Y, w)
		if (old_cost < current_cost):
			print "OUCH! increased cost, returning current weight"
			return w
		costImprovement = ((old_cost - current_cost) / old_cost) * 100.0
		print str(t) + ": " + str(current_cost) + "  diff: " + str(costImprovement)
		v = -(gradient)/np.linalg.norm(gradient)
		w = w + my * v
		t += 1
	print "Finished, returning gradient at a cost improvement of " + format(costImprovement, '.17f')
	return w


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

def classifyDigits():
	initX = append_ones_column_first(images)
	initY = convert_labels(labels)	
	initTheta = np.zeros(785*10).reshape(785,10)
	initTheta[:,0] = 1

	print "running soft_run"
	start = time.clock()
	learnedThetas = soft_run(initX, initY, initTheta)
	end = time.clock()
	print "finished in " + str(end-start) + " seconds"
	np.savez("softmax_matrixOps_learnedThetas_st1_cl01.npz", theta=learnedThetas)



# initY = convert_labels(traininglabels)
# initX = append_ones_column_first(trainingimgs)
# initTheta = np.zeros(785*10).reshape(785,10)
# initTheta[:,:] = 0.00000001
# initTheta[:,0] = 1

# classifyDigits()

# someX = np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]).reshape(10,3)

# print softmax(someX.T)[0]
# print softmax(someX)[1]
# print softmax(someX)[2]
# print softmax(someX)[9]


def recognizeNumber(image, thetas):
	image = np.r_[1, image]
	probs = softmax(np.dot(thetas.T, image))
	mostLikelyDigit = np.argmax(probs)
	return probs, mostLikelyDigit

def compare(start_idx, nr_tests, thetas, images, labels):
	errors = 0
	errorpairs = []
	for i in range(start_idx, start_idx+nr_tests):
		prob, guess = recognizeNumber(images[i], thetas)
		label = labels[i]
		if (guess != label):
			errors += 1
			# print "error! guessing: " + str(guess) + " actual: " + str(label) + "    " + str(prob) + "%"
			# visualize_image(i, testimages)
			# plt.show()
	print "total errors: " + str(errors) + "/" + str(nr_tests)
	pct_error = (float(errors)/nr_tests)*100
	print "error pct: {0:.0f}%".format(pct_error)
	return errors


# myThetas = np.load("softmax_learnedThetas_st1_cl01.npz")['theta']
# compare(0, 10000, myThetas, testimages, testlabels)











datafile = np.load('mnistTrain.npz')

labels = np.squeeze(datafile['labels'])
images = datafile['images']


def softmax(x):
	# x is a K-dimensional vector, K = 10

	# OPFOERER DEN SIG KORREKT??
	# Tager vi amax og expsum for hele matricen x,
	# og udregner derefter noget per indgang i for-loekken??
	# burde den nogensinde give rent 0? (problem med ln senere, ln(0) = -Inf)
	max_x = np.amax(x)
	expsum = np.sum(np.exp(x - max_x))
	return np.array([np.exp(xi - (max_x + np.log(expsum))) for xi in x])

def wtf(X, Y, theta):
	print X.shape
	print Y.shape
	print theta.shape
	tmp = np.dot(X, theta).T
	tmp = softmax(tmp)
	print tmp
	tmp = np.log(tmp)
	tmp = np.dot(Y, tmp)
	tmp = -np.sum(tmp)
	lol = (1.0 / X.shape[0]) * tmp
	# lol = np.log(softmax(np.dot(X, theta).T)).T
	# print np.dot(lol, Y.T)
	# print softmax((np.dot(X, theta))).shape
	# print softmax((np.dot(X, theta).T)).shape
	# print Y.shape

	costsum = 0.0
	for i in range(0, X.shape[0]):
		costsum += np.dot(Y[i], np.log(softmax(np.dot(theta.T, X[i]))))

	costsum = (1.0 / X.shape[0]) * (-costsum)
	print str(costsum)
	print str(lol)

	print "difference: " + str(lol-costsum)
	# lol = np.log(softmax(np.dot(X, theta).T)).T
	# print "will print"
	# print np.dot(lol, Y.T)
	# print "will not print"

print "STARTING"
# initX = append_ones_column_first(images)
# initY = convert_labels(labels)	
# initTheta = np.zeros(785*10).reshape(785,10)
# initTheta[:,0] = 1

wtf_param = 48
initX = np.c_[np.ones(wtf_param), np.arange(wtf_param * 4).reshape(wtf_param,4)]
# print initX.shape
initY = np.arange(wtf_param * 2).reshape(wtf_param, 2)
# print initY.shape
initTheta = np.zeros(10).reshape(5,2)
initTheta[:,0] = 1
# print initTheta.shape

wtf(initX, initY, initTheta)
print "STOPPING"












