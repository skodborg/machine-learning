import numpy as np
import matplotlib.pyplot as plt
import time

smth = np.load('ex1_data.npz')

dat1 = smth['dat1']
target1 = smth['target1']

dat2 = smth['dat2']
target2 = smth['target2']

dat3 = smth['dat3']
target3 = smth['target3']

dat4 = smth['dat4']
target4 = smth['target4']


def plot_points(data, target, figureNo):
	greenX = []
	greenY = []
	redX = []
	redY = []

	# filtering out x0 = 1 as we don't want to plot this
	# data_filtered = [[i[1], i[2]] for i in data]

	for idx, val in enumerate(target):
		if (target[idx] == 1):
			greenX.append(data[idx][0])
			greenY.append(data[idx][1])
		else:
			redX.append(data[idx][0])
			redY.append(data[idx][1])
	plt.figure(figureNo)
	plt.plot(redX, redY, 'ro', greenX, greenY, 'gv')


def plot_linear_regression(data, target, xMin, xMax, figureNo):

	# filtering out x0 = 1 as we don't want to plot this
	data_filtered = [[i[1], i[2]] for i in data]

	# plot the data
	plot_points(data_filtered, target, figureNo)

	regressionX = []
	regressionY = []

	w = np.dot(np.linalg.pinv(data),target);

	# solve w0*x0 + w1*x1 + w2*x2 = 0 for x2 gives
	# x2 = -(w0 + w1*x1)/w2
	# f(x) = -w0/w2 - w1/w2 * x
	b = -w[0]/w[2]
	a = -w[1]/w[2]

	regressionX.append(xMin)
	regressionY.append(a*xMin+b)
	regressionX.append(xMax)
	regressionY.append(a*xMax+b)

	plt.figure(figureNo)
	plt.plot(regressionX, regressionY, 'b-')


def sign(s):
	if (s > 0): return 1
	if (s < 0): return -1
	else: return 1


def h(x, w):
	return sign(np.dot(np.transpose(w), x))


def perceptron(argx, argtarget, argw):
	w = argw
	while(True):
		current_classifications = []

		for idx, x in enumerate(argx):
			# find classifications using current weight vector on input
			curr_h_val = h(x, w)
			current_classifications.append(curr_h_val)

		# find occurences where classifications mismatch known classification
		mismatch_idx = 0
		error_in = 0
		for idx, x in enumerate(current_classifications):
			if (x != argtarget[idx]): 
				error_in += 1
				mismatch_idx = idx
		if (error_in == 0):
			return w
		else:
			# update weight vector
			w = w + argtarget[mismatch_idx] * argx[mismatch_idx]
			# repeat until no mismatches(error_in) on data 
			# (assuming linear separation is possible)


def plot_perceptron(data, target, xMin, xMax, figureNo):

	# filtering out x0 = 1 as we don't want to plot this
	# data_filtered = [[i[1], i[2]] for i in data]
	
	# plot_points(data, target, figureNo)

	perceptronX = []
	perceptronY = []

	# find the hyperplane splitting the data, starting from w = [0,0,0]
	# and iteratively improving until all data is correctly split
	w = perceptron(data, target, [0,0,0])

	# solve w0*x0 + w1*x1 + w2*x2 = 0 for x2 gives
	# x2 = -(w0 + w1*x1)/w2
	# f(x) = -w0/w2 - w1/w2 * x
	b = -w[0]/w[2]
	a = -w[1]/w[2]

	perceptronX.append(xMin)
	perceptronY.append(a*xMin+b)
	perceptronX.append(xMax)
	perceptronY.append(a*xMax+b)

	plt.figure(figureNo)
	plt.plot(perceptronX, perceptronY, 'b-', linewidth=2)

def faster_plot_perceptron(data, target, xMin, xMax, figureNo):
	# filtering out x0 = 1 as we don't want to plot this
	data_filtered = [[i[1], i[2]] for i in data]

	plot_points(data_filtered, target, figureNo)

	perceptronX = []
	perceptronY = []

	# estimate initial weight by help of linear regression to speed up perceptron
	initial_w = np.dot(np.linalg.pinv(data),target);

	# find the hyperplane splitting the data, starting from w = [0,0,0]
	# and iteratively improving until all data is correctly split
	w = perceptron(data, target, initial_w)

	# solve w0*x0 + w1*x1 + w2*x2 = 0 for x2 gives
	# x2 = -(w0 + w1*x1)/w2
	# f(x) = -w0/w2 - w1/w2 * x
	b = -w[0]/w[2]
	a = -w[1]/w[2]

	perceptronX.append(xMin)
	perceptronY.append(a*xMin+b)
	perceptronX.append(xMax)
	perceptronY.append(a*xMax+b)

	plt.figure(figureNo)
	plt.plot(perceptronX, perceptronY, 'b-')



# plot_perceptron(dat1, target1, 0, 1, 1)
# plot_perceptron(dat2, target2, 0, 200, 2)
# plot_perceptron(dat3, target3, 0, 200, 3)

# start = time.clock()
# plot_perceptron(dat4, target4, 0, 200, 4)
# end = time.clock()
# print "Original perceptron algorithm: ", end - start
# start = time.clock()
# faster_plot_perceptron(dat4, target4, 0, 200, 5)
# end = time.clock()
# print "Faster perceptron algorithm: ", end - start

# plot_linear_regression(dat1, target1, 0, 1, 2)
# plot_linear_regression(dat2, target2, 0, 200, 2)
# plot_linear_regression(dat3, target3, 0, 200, 3)
# plot_linear_regression(dat4, target4, 0, 200, 4)

# plt.show()

