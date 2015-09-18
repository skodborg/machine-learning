import numpy as np
import matplotlib.pyplot as plt
import math
import time

datafile = np.load('mnistTrain.npz')

labels = np.squeeze(datafile['labels'])
images = datafile['images']

testdatafile = np.load('mnistTest.npz')

testlabels = np.squeeze(testdatafile['labels'])
testimages = testdatafile['images']



def visualize_image(imageidx, images):
	image = images[imageidx]
	# assuming quadratic spaces containing the images
	size = math.sqrt(image.shape[0])
	# generating x,y axes
	x = np.linspace(0, size-1, size)
	y = np.linspace(0, size-1, size)
	# preparing (28,28)-matrix containing pixel-values for plotting
	image = np.rot90(np.reshape(np.array(image), (28,28)))
	# plotting
	plt.pcolormesh(x, y, image)		

def logistic_func(z):
	return 1 / (1 + np.exp(-z))

def weight_sum(theta, x):
	return np.dot(np.transpose(theta), x)

def h(theta, x):
	return logistic_func(weight_sum(theta, x))


theta0 = np.load('s01_p0001_learnedTheta_digit_0.npz')['theta']
theta1 = np.load('s01_p0001_learnedTheta_digit_1.npz')['theta']
theta2 = np.load('s01_p0001_learnedTheta_digit_2.npz')['theta']
theta3 = np.load('s01_p0001_learnedTheta_digit_3.npz')['theta']
theta4 = np.load('s01_p0001_learnedTheta_digit_4.npz')['theta']
theta5 = np.load('s01_p0001_learnedTheta_digit_5.npz')['theta']
theta6 = np.load('s01_p0001_learnedTheta_digit_6.npz')['theta']
theta7 = np.load('s01_p0001_learnedTheta_digit_7.npz')['theta']
theta8 = np.load('s01_p0001_learnedTheta_digit_8.npz')['theta']
theta9 = np.load('s01_p0001_learnedTheta_digit_9.npz')['theta']


trainingimgs = images[0:45000,]
traininglabels = labels[0:45000,]
validationimgs = images[45000:60000,]
validationlabels = labels[45000:60000,]


def getThetaForDigit(digit):
	thetas = [theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9]
	return thetas[digit]


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


# compare(0,10000)



def approximate_gradient(f, x, eps):
	return (f(x+eps) - f(x-eps))/(2*eps)

# print approximate_gradient(lambda x: x*x, 10, 0.0001)
# prints 20.0 as expected


def log_cost(X, y, theta, lambda_i):
	def logistic_func(z):
		return 1 / (1 + np.exp(-z))

	def weight_sum(theta, x):
		return np.dot(np.transpose(theta), x)

	def h(theta, x):
		return logistic_func(weight_sum(theta, x))

	lmb = 75
	# regularization_param = (lmb**lambda_i) * np.sum(theta[1:,]**2)
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
	my = 0.05
	current_cost = 1000000.0
	costImprovLimit = 0.001
	costImprovement = 1.0
	t = 0
	# for t in range(0, 10000):
	while costImprovement > costImprovLimit:
		costfn, gradient = log_cost(X, y, w, lambda_i)
		old_cost = current_cost
		current_cost = costfn(X, y, w)
		if (old_cost < current_cost):
			print "OUCH! increased cost, returning current weight"
			return w
		costImprovement = ((old_cost - current_cost) / old_cost) * 100.0
		# print str(t) + ": " + str(current_cost) + "  diff: " + str(costImprovement)
		v = -(gradient)/np.linalg.norm(gradient)
		w = w + my * v
		t += 1
	print "Finished, returning gradient at a cost improvement of " + format(costImprovement, '.17f')
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


def findBestRegularizedTheta():
	images = validationimgs
	labels = validationlabels
	best_model = 0
	lowest_errors = 10000000
	for i in range(-6,6):
		thetas = []
		for j in range(0,10):
			currTheta = findThetaForClassifier(j, i, images, labels)
			thetas.append(currTheta)
		curr_error = compare(0, 15000, thetas, images, labels)
		if (curr_error < lowest_errors):
			best_model = i
			lowest_errors = curr_error
			print "new best: " + str(i) + " with " + str(curr_error) + " errors"
	print "done with i = " + str(best_model)


# findBestRegularizedTheta()
	





def classifyDigits():
	for i in range(0,10):
		print "learning digit " + str(i)
		start = time.clock()
		learnedTheta = findThetaForClassifier(i, 1, images, labels)
		end = time.clock()
		print "learned in " + str(end-start) + " seconds"
		np.savez("st_05_cl_001_learnedTheta_digit_"+str(i)+".npz", theta=learnedTheta)

classifyDigits()

