import numpy as np
import matplotlib.pyplot as plt
import his_ex2 as ex2
import math

datafile = np.load('mnistTrain.npz')

print datafile.files

# labels.shape = (60000,1)
labels = datafile['labels']
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


def log_cost(X, Y, theta):
	def logistic_func(z):
		return 1 / (1 + np.exp(-z))

	def weight_sum(theta, X):
		return np.dot(np.transpose(theta), X)

	def h(theta, X):
		return logistic_func(weight_sum(theta, X))

	fn_cost = lambda X, Y, theta: -np.sum([y * np.log(h(theta, x)) 
		+ (1 - y) * np.log(1 - h(theta, x)) for x,y in zip(X,Y)])
	gradient = np.dot(np.transpose(-X), (Y - logistic_func(np.dot(X, theta))))
	return fn_cost, gradient


X = np.array([[1,2,3],
			  [1,2,3],
			  [1,2,3],
			  [1,2,3],
			  [1,2,3],
			  [1,2,3]])
Y = np.array([7,3,9,0,4,2])
theta = np.array([2,2,1])

costfn, gradient = log_cost(X, Y, theta)
# print type(costfn)
# print type(gradient)
# print costfn(X, Y, theta)
# print gradient



