import numpy as np
import matplotlib.pyplot as plt
import his_ex2 as ex2
import linreg_and_perceptron as lp
import math

datafile = np.load('mnistTrain.npz')

print datafile.files

# labels.shape = (60000,1)
labels = np.squeeze(datafile['labels'])
# images.shape = (60000,784)
images = datafile['images']


def visualize_image(imageidx):
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


# for i in range(0,10):
# 	plt.figure(i)
# 	visualize_image(i)
# 	print str(i) + str(labels[i])
# plt.show()


def approximate_gradient(f, x, eps):
	return (f(x+eps) - f(x-eps))/(2*eps)

# print approximate_gradient(lambda x: x*x, 10, 0.0001)
# prints 20.0 as expected


def log_cost(X, y, theta):
	def logistic_func(z):
		return 1 / (1 + np.exp(-z))

	def weight_sum(theta, x):
		return np.dot(np.transpose(theta), x)

	def h(theta, x):
		return logistic_func(weight_sum(theta, x))

	# ------------- cost function using matrix operations (faster) ------------
	fn_cost = lambda X, Y, theta: -np.sum(y * np.log(logistic_func(np.dot(X, theta))) 
		+ (1 - y) * np.log(1 - logistic_func(np.dot(X, theta))))

	# ------------- cost function using a for-loop ----------------------------
	# fn_cost = lambda X, Y, theta: -np.sum([y * np.log(h(theta, x)) 
	# 	+ (1 - y) * np.log(1 - h(theta, x)) for x,y in zip(X,Y)])

	gradient = np.dot(np.transpose(-X), (y - logistic_func(np.dot(X, theta))))
	return fn_cost, gradient

def log_grad(X, y, theta):
	# initialize w(0) and v(0)
	w = theta
	v = np.zeros(1)
	my = 0.0000001
	current_cost = 1000000
	costImprovLimit = 0.01
	costImprovement = 1
	t = 0
	# for t in range(0, 10000):
	while costImprovement > costImprovLimit:
	# while current_cost > 5.5:
		# pull gradient
		costfn, gradient = log_cost(X, y, w)
		old_cost = current_cost
		current_cost = costfn(X, y, w)
		costImprovement = ((old_cost - current_cost) / old_cost) * 100
		print str(t) + ": " + str(current_cost) + "  diff: " + str(costImprovement)
		v = -(gradient)
		w = w + my * v
		t += 1
	return w



def findThetaForClassifier(digit):
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
	learnedTheta = log_grad(inputX, inputY, initTheta)

	return learnedTheta


def classifyDigits():
	for i in range(0,10):
		print "learning digit " + str(i)
		learnedTheta = findThetaForClassifier(i)
		np.savez("learnedTheta_digit_"+str(i)+".npz", theta=learnedTheta)



classifyDigits()

