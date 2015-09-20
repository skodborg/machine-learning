import numpy as np
import matplotlib.pyplot as plt
import math
import time
import random

datafile = np.load('mnistTrain.npz')
testdatafile = np.load('mnistTest.npz')
autraindatafile = np.load('auTrainMerged.npz')

labels = np.squeeze(datafile['labels'])
images = datafile['images']

testlabels = np.squeeze(testdatafile['labels'])
testimages = testdatafile['images']

autrainlabels = autraindatafile['labels']
autrainimages = autraindatafile['digits']

pos_auTwos = (autrainlabels == 2)
pos_auSevens = (autrainlabels == 7)
auTwos = autrainimages[pos_auTwos]
auSevens = autrainimages[pos_auSevens]
auTwosSevens = np.concatenate([auTwos, auSevens])
auTwosSevensLabels = np.concatenate([np.ones(auTwos.shape[0]), np.zeros(auSevens.shape[0])])


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
	plt.pcolormesh(x, y, image, cmap=plt.cm.gray)


def logistic_func(z):
	return 1 / (1 + np.exp(-z))

def weight_sum(theta, x):
	return np.dot(np.transpose(theta), x)

def h(theta, x):
	return logistic_func(weight_sum(theta, x))


theta0 = np.load('unreg_cl_001_learnedTheta_digit_0.npz')['theta']
theta1 = np.load('unreg_cl_001_learnedTheta_digit_1.npz')['theta']
theta2 = np.load('unreg_cl_001_learnedTheta_digit_2.npz')['theta']
theta3 = np.load('unreg_cl_001_learnedTheta_digit_3.npz')['theta']
theta4 = np.load('unreg_cl_001_learnedTheta_digit_4.npz')['theta']
theta5 = np.load('unreg_cl_001_learnedTheta_digit_5.npz')['theta']
theta6 = np.load('unreg_cl_001_learnedTheta_digit_6.npz')['theta']
theta7 = np.load('unreg_cl_001_learnedTheta_digit_7.npz')['theta']
theta8 = np.load('unreg_cl_001_learnedTheta_digit_8.npz')['theta']
theta9 = np.load('unreg_cl_001_learnedTheta_digit_9.npz')['theta']
thetas = [theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9]

# np.savez("all_vs_ones_weights.npz", thetas=thetas)


def recognizeNumber(image, thetas):
	image = np.r_[1, image]
	maxProb = 0
	mostLikelyDigit = 0
	for i in range(0,10):
		currTheta = thetas[i]
		prob = h(currTheta, image)
		if (prob > maxProb):
			maxProb = prob
			mostLikelyDigit = i
	return maxProb, mostLikelyDigit

def recognizeTwos(image, theta):
	image = np.r_[1, image]
	prob = h(theta.T, image)
	result = 0.0
	if (prob > 0.5):
		result = 1.0
	return prob, result


def approximate_gradient(f, x, eps):
	return (f(x+eps) - f(x-eps))/(2*eps)


def log_cost(X, y, theta, lambda_i):
	def logistic_func(z):
		return 1 / (1 + np.exp(-z))

	def weight_sum(theta, x):
		return np.dot(np.transpose(theta), x)

	def h(theta, x):
		return logistic_func(weight_sum(theta, x))

	# regularization using weight decay as regularizer and a varying parameter, 
	# given by 3 to the power of lambda_i   (3^i * wTw)
	lmb = 3
	# regularization_param = (lmb**lambda_i) * np.sum(theta[1:,]**2)
	# regularization_param = (lmb**lambda_i) * np.dot(theta[1:,].T, theta[1:,])
	regularization_param = 0
	# regularization_param_derivative = (lmb**lambda_i) * np.sum(2 * theta[1:,])
	regularization_param_derivative = 0

	# ------------- cost function including regularization --------------------
	fn_cost = lambda X, Y, theta: (1.0 / X.shape[0]) * (-np.sum(y * np.log(logistic_func(np.dot(X, theta))) 
		+ (1 - y) * np.log(1 - logistic_func(np.dot(X, theta))))) + regularization_param

	# ------------- cost function using matrix operations (faster) ------------
	# fn_cost = lambda X, Y, theta: -np.sum(y * np.log(logistic_func(np.dot(X, theta))) 
	# 	+ (1 - y) * np.log(1 - logistic_func(np.dot(X, theta))))

	# ------------- cost function using a for-loop ----------------------------
	# fn_cost = lambda X, Y, theta: -np.sum([y * np.log(h(theta, x)) 
	# 	+ (1 - y) * np.log(1 - h(theta, x)) for x,y in zip(X,Y)])

	# gradient = np.dot(np.transpose(-X), (y - logistic_func(np.dot(X, theta))))
	gradient = (1.0 / X.shape[0]) * np.dot(np.transpose(-X), 
		(y - logistic_func(np.dot(X, theta)))) + regularization_param_derivative
	return fn_cost, gradient


def log_grad(X, y, theta, lambda_i):
	w = theta
	v = np.zeros(1)
	my = 0.025
	costImprovLimit = 0.001
	current_cost = 1000000.0
	costImprovement = 1.0
	t = 0
	iters = []
	costs = []
	# for t in range(0, 10000):
	while costImprovement > costImprovLimit:
		costfn, gradient = log_cost(X, y, w, lambda_i)
		old_cost = current_cost
		current_cost = costfn(X, y, w)
		# plot cost per iteration
		# costs.append(current_cost)
		# iters.append(t)
		if (old_cost < current_cost):
			print "OUCH! increased cost, returning current weight with latest cost impr. of " + str(costImprovement)
			# plot costs per iteration
			# plt.plot(iters, costs, 'r-')
			# plt.show()
			return w
		costImprovement = ((old_cost - current_cost) / old_cost) * 100.0
		# print str(t) + ": " + str(current_cost) + "  diff: " + str(costImprovement)
		v = -(gradient)/np.linalg.norm(gradient)
		w = w + my * v
		# plot weight after 5 iterations
		# if (t == 5):
			# plt.imshow(w[1:,].reshape(28,28))
			# plt.show()
		t += 1
	print "Finished, returning gradient at a cost improvement of " + format(costImprovement, '.17f')
	# plot costs per iteration
	# plt.plot(iters, costs, 'r-')
	# plt.show()
	return w


def findThetaForClassifier(digit, lambda_i, images, labels):
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
	learnedTheta = log_grad(inputX, inputY, initTheta, lambda_i)

	return learnedTheta



def compare(start_idx, nr_tests, thetas, images, labels):
	errors = 0
	errorpairs = []
	for i in range(start_idx, start_idx+nr_tests):
		prob, guess = recognizeNumber(images[i], thetas)
		# prob, guess = recognizeTwos(images[i], thetas)
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

# results:
# for lambda = 3^i  i = {-6, ..., 5}
#   best result was 1331 errors for i = -6
# for lambda = 0
#   result was 1310 errors
# for lambda = 2^i  i = {-6, ..., 5}
#   best result was 1356 errors for i = -5
# for lambda = 4^i  i = {-6, ..., 5}
#   best result was 1315 errors for i = -6
# for lambda = 10^i i = {-6, ..., 5}
#   best result was 1308 errors for i = -5

# BEST REGULARIZING PARAMETER SO FAR: -35 with 1254 errors on validation set
# -40 gave 1318 errors
# -35 & -39 gave 1254 errors
# -30 gave 1275 errors
# -100 gave 1298 errors
# -10 gave 1352 errors

def findBestRegularizedTheta(trainingimgs, validationimgs, traininglabels, validationlabels):
	validimgs = validationimgs
	validlbls = validationlabels
	trainimgs = trainingimgs
	trainlbls = traininglabels
	best_model = 0
	lowest_errors = 10000000
	for i in range(-6,6):
		thetas = []
		for j in range(0,10):
			print "training digit " + str(j) + " with regularization param: " + str(i)
			currTheta = findThetaForClassifier(j, i, trainimgs, trainlbls)
			thetas.append(currTheta)
		curr_error = compare(0, validimgs.shape[0], thetas, validimgs, validlbls)
		if (curr_error < lowest_errors):
			best_model = i
			lowest_errors = curr_error
			print "new best: " + str(i) + " with " + str(curr_error) + " errors"
	print "done with i = " + str(best_model)


def learnTwosVsSevens():
	trainingimgs, validationimgs, traininglabels, validationlabels = training_validation_split(auTwosSevens, auTwosSevensLabels, 5)
	best_theta = findThetaForClassifier(1, 1, trainingimgs, traininglabels)
	# best_theta = np.load("params.npz")['theta']
	# np.savez("params.npz", theta=best_theta)
	print "in-sample error on training set:"
	compare(0, auTwosSevens.shape[0], np.array([best_theta]), auTwosSevens, auTwosSevensLabels)
	print "out-of-sample error on validation set:"
	compare(0, validationimgs.shape[0], np.array([best_theta]), validationimgs, validationlabels)


def plotAlVsOnesWeights():
	thetas = np.load('all_vs_ones_weights.npz')['thetas']
	for theta in thetas:
		plt.imshow(np.rot90(theta[1:,].reshape(28,28)))
		plt.show()


def classifyDigits():
	for i in range(7,8):
		print "learning digit " + str(i)
		start = time.clock()
		learnedTheta = findThetaForClassifier(i, -35, images, labels)
		end = time.clock()
		print "learned in " + str(end-start) + " seconds"
		np.savez("unreg_cl_001_learnedTheta_digit_"+str(i)+".npz", theta=learnedTheta)


# for i in range(0,10):
# 	visualize_image(i, images)
# 	print "prediction: " + str(recognizeNumber(images[i], thetas))
# 	plt.show()

# # in-sample
# compare(0, 60000, thetas, images, labels)
# # out-of-sample
# compare(0, 10000, thetas, testimages, testlabels)

# learnTwosVsSevens()
# plotAlVsOnesWeights()
# classifyDigits()
