import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random

def training_validation_split(aSetImgs, aSetLabels, ratio):
	comb = np.c_[aSetImgs, aSetLabels]
	np.random.shuffle(comb)
	validation_size = aSetImgs.shape[0]/ratio
	training_size = aSetImgs.shape[0] - validation_size
	training_set = comb[0:training_size,:]
	validation_set = comb[training_size:aSetImgs.shape[0],:]

	training_imgs = training_set[:,0:training_set.shape[1]-1]
	validation_imgs = validation_set[:,0:validation_set.shape[1]-1]

	training_labels = training_set[:,training_set.shape[1]-1]
	validation_labels = validation_set[:,validation_set.shape[1]-1]
	return training_imgs, validation_imgs, training_labels, validation_labels


def visualize_image(imageidx, images):
	image = images[imageidx]
	# assuming quadratic spaces containing the images
	size = math.sqrt(image.shape[0])
	# generating x,y axes
	x = np.linspace(0, size-1, size)
	y = np.linspace(0, size-1, size)
	# preparing (28,28)-matrix containing pixel-values for plotting
	image = np.rot90(np.reshape(np.array(image), (28,28)))
	# image = np.reshape(np.array(image), (28,28)).T
	# plotting
	plt.pcolormesh(x, y, image)

# operates on a vector
def softmax(x):
	max_x = np.amax(x)
	expsum = np.sum(np.exp(x - max_x))
	return np.exp(x - (max_x + np.log(expsum)))

# applies softmax(x) on each vector in matrix-argument X
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

def soft_cost(X, Y, theta, useRegularization=False, lambda_i=0):
	fn_cost = -(np.sum(Y * np.log(matrix_softmax(np.dot(X, theta)))))
	gradient = np.dot(-X.T, (Y - matrix_softmax(np.dot(X, theta))))
	return fn_cost, gradient

# def old_soft_run(X, Y, theta, useRegularization=False, lambda_i=0):
# 	w = theta
# 	my = 0.2
# 	current_cost = 100000000.0
# 	costImprovLimit = 0.001
# 	costImprovement = 1.0
# 	iters = []
# 	costs = []
# 	t = 0
# 	# interactive plotting as we progress
# 	plt.ion()
# 	plt.show()
# 	for t in range(0, 1000):
# 	# while costImprovement > costImprovLimit:
# 		costfn, gradient = soft_cost(X, Y, w)
# 		gradient = (1.0 / X.shape[0]) * gradient
# 		old_cost = current_cost
# 		current_cost = (1.0 / X.shape[0]) * costfn
# 		# plot cost per iteration
# 		costs.append(current_cost)
# 		iters.append(t)

# 		if (old_cost < current_cost):
# 			print "OUCH! increased cost, returning current weight. Latest cost impr. of " + str(costImprovement) + " using step size " + str(my)
# 			print "cost at: " + str(current_cost)
# 			# plot costs per iteration
# 			# if (current_cost > 0.15):
# 				# plt.plot(iters, costs, 'r-')
# 				# plt.show()
# 			return w
# 		costImprovement = ((old_cost - current_cost) / old_cost) * 100.0
		
# 		print str(t) + ": " + str(current_cost) + " diff: " + str(costImprovement)

# 		v = -(gradient)
# 		w = w + my * v

# 		# interactive plotting as we progress
# 		plt.plot(iters, costs, 'r-')
# 		plt.draw()
# 		if (t % 100 == 0):
# 			compare(0, validationimgs.shape[0], w, validationimgs, validationlabels)
# 		t += 1
# 	return w


def logistic_func(z):
	return 1 / (1 + np.exp(-z))

def weight_sum(theta, x):
	return np.dot(np.transpose(theta), x)

def h(theta, x):
	return logistic_func(weight_sum(theta, x))


def recognizeNumber(image, thetas):
	image = np.r_[1, image]
	maxProb = 0
	mostLikelyDigit = 0
	for i in range(0,10):
		currTheta = thetas[i]
		print currTheta.shape
		print image.shape
		prob = h(currTheta, image)
		if (prob > maxProb):
			maxProb = prob
			mostLikelyDigit = i
	return maxProb, mostLikelyDigit

def recognizeNumber_softmax(image, thetas):
	image = np.r_[1, image]
	probs = softmax(np.dot(thetas.T, image))
	mostLikelyDigit = np.argmax(probs)
	return probs, mostLikelyDigit

def recognizeTwos(image, theta):
	image = np.r_[1, image]
	prob = h(theta.T, image)
	result = 0.0
	if (prob > 0.5):
		result = 1.0
	return prob, result


def approximate_gradient(f, x, eps):
	return (f(x+eps) - f(x-eps))/(2*eps)


def log_cost(X, y, theta, lambda_i=0, useRegularization=False):
	if (useRegularization):
		# regularization using weight decay as regularizer and a varying parameter, 
		# given by 3 to the power of lambda_i   (3^i * wTw)
		lmb = 3
		regularization_param = (lmb**lambda_i) * np.sum(theta[1:,]**2)
		regularization_param_derivative = (lmb**lambda_i) * np.sum(2 * theta[1:,])
	else:
		regularization_param = 0
		regularization_param_derivative = 0

	# ------------- cost function including regularization --------------------
	fn_cost = -np.sum(y * np.log(logistic_func(np.dot(X, theta))) 
		+ (1 - y) * np.log(1 - logistic_func(np.dot(X, theta)))) + regularization_param

	# ------------- cost function using matrix operations (faster) ------------
	# fn_cost = -np.sum(y * np.log(logistic_func(np.dot(X, theta))) 
	# 	+ (1 - y) * np.log(1 - logistic_func(np.dot(X, theta))))

	# ------------- cost function using a for-loop ----------------------------
	# fn_cost = -np.sum([y * np.log(h(theta, x)) 
	# 	+ (1 - y) * np.log(1 - h(theta, x)) for x,y in zip(X,Y)])

	gradient = np.dot(np.transpose(-X), 
		(y - logistic_func(np.dot(X, theta)))) + regularization_param_derivative
	return fn_cost, gradient


def log_grad(X, y, theta, lambda_i=0, useRegularization=False, plotWeights=False, plotCosts=False):
	stepSize = 0.1
	return gradient_descent(X, y, theta, log_cost, lambda_i, useRegularization, plotWeights, plotCosts, stepSize)

def soft_run(X, Y, theta, lambda_i=0, useRegularization=False, plotWeights=False, plotCosts=False):
	stepSize = 0.05
	return gradient_descent(X, Y, theta, soft_cost, lambda_i, useRegularization, plotWeights, plotCosts, stepSize)

def gradient_descent(X, y, theta, fn_cost, lambda_i, useRegularization, plotWeights, plotCosts, stepSize):
	w = theta
	my = stepSize
	costImprovLimit = 0.001
	current_cost = 1000000.0
	costImprovement = 1.0
	t = 0

	# plot cost per iteration as we go
	if (plotCosts):
		plt.figure(0)
		plt.ion()
		plt.show()
		iters = []
		costs = []

	# for t in range(0, 500):
	while costImprovement > costImprovLimit:
		costfn, gradient = fn_cost(X, y, w, lambda_i, useRegularization)
		gradient = (1.0 / X.shape[0]) * gradient
		old_cost = current_cost
		current_cost = (1.0 / X.shape[0]) * costfn
		
		if (old_cost < current_cost):
			# print "OUCH! increased cost, returning current weight. Latest cost impr. of " + str(costImprovement) + " using step size " + str(my)
			# print "cost at: " + str(current_cost)
			# plot costs per iteration
			# if (current_cost > 0.15):
				# plt.plot(iters, costs, 'r-')
				# plt.show()
			return w
		costImprovement = ((old_cost - current_cost) / old_cost) * 100.0
		# print str(t) + ": " + str(current_cost) + "  diff: " + str(costImprovement)
		v = -(gradient)/np.linalg.norm(gradient)
		w = w + my * v
		# plot weight after 5 iterations
		if (t == 5 and plotWeights):
			plt.figure(1)
			plt.imshow(w[1:,].reshape(28,28))
			plt.show()
			plt.figure(0)
		# plot cost per iteration as we go
		if (plotCosts):
			costs.append(current_cost)
			iters.append(t)
			plt.plot(iters, costs, 'r-')
			plt.draw()
		t += 1
	# print "Finished, returning gradient at a cost improvement of " + format(costImprovement, '.17f')
	return w	


def findThetaForClassifier(digit, lambda_i, images, labels, useRegularization=False, plotWeights=False):
	pos_of_digit = (labels == digit)
	pos_of_others = (labels != digit)

	# find all entries in data labeled with the digit
	myDigitsX = images[pos_of_digit]
	# add x0 bias term to x's
	myDigitsX = np.c_[np.ones(myDigitsX.shape[0]), myDigitsX]

	# add all-1's label vector (we're classifying this digit as true)
	myDigitsY = np.ones(myDigitsX.shape[0])

	# find all entries in data that is not the digit we want
	otherDigitsX = images[pos_of_others]
	otherDigitsX = np.c_[np.ones(otherDigitsX.shape[0]), otherDigitsX]

	# add all-0's label vector (we're training to recognize occurences of 'digit' from
	# everything else, otherDigitsX being everything else)
	otherDigitsY = np.zeros(otherDigitsX.shape[0])

	# combine the data sets
	inputX = np.concatenate([myDigitsX, otherDigitsX])
	inputY = np.concatenate([myDigitsY, otherDigitsY])
	initTheta = np.zeros(inputX.shape[1])

	# use logistic gradient descent to minimize errors and return
	# a learned theta when errors is at a tolerable level
	learnedTheta = log_grad(inputX, inputY, initTheta, lambda_i, useRegularization, plotWeights)

	return learnedTheta



def compare(start_idx, nr_tests, thetas, images, labels, only2vs7=False, softmax=False):
	errors = 0
	errorpairs = []
	for i in range(start_idx, start_idx+nr_tests):
		if (only2vs7):
			prob, guess = recognizeTwos(images[i], thetas)
		elif (softmax):
			prob, guess = recognizeNumber_softmax(images[i], thetas)
		else:
			prob, guess = recognizeNumber(images[i], thetas)
		label = labels[i]
		if (guess != label):
			errors += 1
			# print "error! guessing: " + str(guess) + " actual: " + str(label) + "    " + str(prob) + "%"
			# visualize_image(i, images)
			# plt.show()
	print "total errors: " + str(errors) + "/" + str(nr_tests)
	pct_error = (float(errors)/nr_tests)*100
	print "error pct: {0:.0f}%".format(pct_error)
	return errors


def findBestRegularizedTheta(trainingimgs, validationimgs, traininglabels, validationlabels):
	validimgs = validationimgs
	validlbls = validationlabels
	trainimgs = trainingimgs
	trainlbls = traininglabels
	best_model = 0
	lowest_errors = 10000000
	# for i in [-30, -10, -5, 0, 5, 10, 30]:
	for i in range(-6,6):
		thetas = []
		for j in range(0,10):
			print "training digit " + str(j) + " with regularization param: " + str(i)
			currTheta = findThetaForClassifier(j, i, trainimgs, trainlbls, True)
			thetas.append(currTheta)
		curr_error = compare(0, validimgs.shape[0], thetas, validimgs, validlbls)
		if (curr_error < lowest_errors):
			best_model = i
			lowest_errors = curr_error
			print "new best: " + str(i) + " with " + str(curr_error) + " errors"
	print "done with i = " + str(best_model)


def learnTwosVsSevens():
	# loading relevant data
	autraindatafile = np.load('auTrainMerged.npz')
	autrainlabels = autraindatafile['labels']
	autrainimages = autraindatafile['digits']
	auTwos = autrainimages[(autrainlabels == 2)]
	auSevens = autrainimages[(autrainlabels == 7)]
	auTwosSevens = np.concatenate([auTwos, auSevens])
	auTwosSevensLabels = np.concatenate([np.ones(auTwos.shape[0]), np.zeros(auSevens.shape[0])])

	trainingimgs, validationimgs, traininglabels, validationlabels = training_validation_split(auTwosSevens, auTwosSevensLabels, 5)
	best_theta = findThetaForClassifier(1, 1, trainingimgs, traininglabels)
	# best_theta = np.load("params.npz")['theta']
	# np.savez("params.npz", theta=best_theta)
	print "in-sample error on training set:"
	compare(0, trainingimgs.shape[0], best_theta, trainingimgs, traininglabels, True)
	print "out-of-sample error on validation set:"
	compare(0, validationimgs.shape[0], best_theta, validationimgs, validationlabels, True)


def plotAlVsOnesWeights():
	thetas = np.load('all_vs_ones_weights.npz')['thetas']
	for theta in thetas:
		plt.imshow(np.rot90(theta[1:,].reshape(28,28)))
		plt.show()


def classifyDigits(images, labels):
	thetas = []
	for i in range(0,10):
		print "learning digit " + str(i)
		start = time.clock()
		learnedTheta = findThetaForClassifier(i, 0, images, labels)
		end = time.clock()
		thetas.append(learnedTheta)
		print "learned in " + str(end-start) + " seconds"
		# np.savez("reg_-7_cl_001_learnedTheta_digit_"+str(i)+".npz", theta=learnedTheta)
	return thetas

def classifyDigits_softmax(images, labels):
	theta = np.zeros(images.shape[1]*labels.shape[1]).reshape(images.shape[1], labels.shape[1])
	print "running soft_run"
	start = time.clock()
	learnedThetas = soft_run(images, labels, theta)
	end = time.clock()
	print "finished in " + str(end-start) + " seconds"
	return learnedThetas


def load_and_estimate_errors():
	datafile = np.load('mnistTrain.npz')
	testdatafile = np.load('mnistTest.npz')
	autraindatafile = np.load('auTrainMerged.npz')

	labels = np.squeeze(datafile['labels'])
	images = datafile['images']

	testlabels = np.squeeze(testdatafile['labels'])
	testimages = testdatafile['images']

	auimgs = autraindatafile['digits']
	aulbls = autraindatafile['labels']

	unreg_thetas = np.load('unreg_all_vs_ones_weights.npz')['thetas']
	# print "MNIST in-sample error:"
	# compare(0, images.shape[0], unreg_thetas, images, labels)
	# print "MNIST out-of-sample error (using test set):"
	# compare(0, testimages.shape[0], unreg_thetas, testimages, testlabels)
	
	# print "\nauDigits 2 vs 7 in-sample and out-of-sample"
	# learnTwosVsSevens()

	# someDigit = 3
	# print "\nPlot of weight on auDigits for digit " + str(someDigit) + " after 5 iterations"
	# findThetaForClassifier(someDigit, 0, auimgs, aulbls, plotWeights=True)
	
	print "Running softmax on auDigits"
	myThetas = classifyDigits_softmax(append_ones_column_first(auimgs), convert_labels(aulbls))
	compare(0, auimgs.shape[0], myThetas, auimgs, aulbls, softmax=True)


if __name__ == "__main__":
    load_and_estimate_errors()
