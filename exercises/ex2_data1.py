import numpy as np
import matplotlib.pyplot as plt
import linreg_and_perceptron as ex1

smth = np.load('ex2_data3.npz')
print smth.files

data = smth['x']
target = smth['y']

# plot data on figure 1
# ex1.plot_points(data, target, 1)


def phi():
	x = data[:,0]
	y = data[:,1]
	def phi_helper(x, y):
		component1 = x**0 * y**0
		component2 = x**0 * y**1
		component3 = x**0 * y**2
		component4 = x**0 * y**3
		component5 = x**1 * y**0
		component6 = x**1 * y**1
		component7 = x**1 * y**2
		component8 = x**2 * y**0
		component9 = x**2 * y**1
		component10 = x**3 * y**0
		return [component1, component2, component3, component4, component5,
			    component6, component7, component8, component9, component10]
	print str(x[0]) + " " + str(y[0])
	print phi_helper(x[0], y[0])

	result = np.zeros(shape=(data.shape[0], 10))
	for idx, e in enumerate(data):
		result[idx] = phi_helper(e[0],e[1])
		# result.append(phi_helper(x,y))

	ones = np.ones(result.shape[0])


	leastErrors = result.shape[0]+1
	leastErrorsWeight = [0,0,0]
	curr_i = -1
	curr_j = -1

	for i in range(0,10):
		for j in range(0,10):
			xs = result[:,i]
			ys = result[:,j]
			fuck = np.c_[ones, xs, ys]

			
			it = 3000 #iterations
			w = [0,0,0] #initial weight
			while(it > 0):
				current_classifications = []

				for idx, x in enumerate(fuck):
					# find classifications using current weight vector on input
					curr_h_val = ex1.h(x, w)
					current_classifications.append(curr_h_val)

				# find occurences where classifications mismatch known classification
				mismatch_idx = 0
				error_in = 0
				for idx, x in enumerate(current_classifications):
					if (x != target[idx]): 
						error_in += 1
						mismatch_idx = idx
				if (error_in == 0):
					leastErrorsWeight = w
					leastErrors = 0
					it = 400
				else:
					# update weight vector
					if (error_in < leastErrors):
						leastErrorsWeight = w
						leastErrors = error_in
						curr_i = i
						curr_j = j
					w = w + target[mismatch_idx] * fuck[mismatch_idx]
					it -= 1
					# repeat until no mismatches(error_in) on data 
					# (assuming linear separation is possible)	

	print "leastErrors: " + str(leastErrors)
	print "with weight: " + str(leastErrorsWeight)
	print "and i,j : " + str(curr_i) + ":" + str(curr_j)


	# ex1_data1
	# best result (3000 iterations on perceptron):
	# leastErrors: 92
	# with weight: [-1.          5.11094781  9.75392334]
	# and i,j : 3:7

	# ex1_data2
	# leastErrors: 188
	# with weight: [-1.          3.60835684 -3.20045259]
	# and i,j : 5:9

	# ex1_data3
	# leastErrors: 160
	# with weight: [ 1.          4.2479373  -5.07218892]
	# and i,j : 4:9

def plot_w():
	regressionX = []
	regressionY = []

	# weight from best result on ex2_data1 with 92 errors
	w = [1, 4.2479373, -5.07218892]

	b = -w[0]/w[2]
	a = -w[1]/w[2]

	regressionX.append(-1)
	regressionY.append(a*(-1)+b)
	regressionX.append(1)
	regressionY.append(a*1+b)

	plt.figure(1)
	plt.plot(regressionX, regressionY, 'b-')

plot_w()

# print "---------- EX2_DATA1 ----------"
# phi()

# smth = np.load('ex2_data2.npz')
# print smth.files

# data = smth['x']
# target = smth['y']

# print "---------- EX2_DATA2 ----------"
# phi()


# smth = np.load('ex2_data3.npz')
# print smth.files

# data = smth['x']
# target = smth['y']

# print "---------- EX2_DATA3 ----------"
# phi()


ourxs = data[:,0]
ourys = data[:,1]

plotxs = ourxs
plotys = ourxs * ourxs * ourxs

plotpointsdata = np.c_[plotxs, plotys]

ex1.plot_points(plotpointsdata, target, 1)

# plt.xlim(-1,1)
# plt.ylim(0,1)
# plt.show()




def iterations_perceptron(argx, argtarget, argw, iterations):
	i = iterations
	w = argw
	while(i > 0):
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
			i -= 1
			# repeat until no mismatches(error_in) on data 
			# (assuming linear separation is possible)	




# plotData = phi(0,0)
# ex1.plot_points(plotData, target, 1)
# plotData = phi(0,1)
# ex1.plot_points(plotData, target, 2)
# plotData = phi(0,2)
# ex1.plot_points(plotData, target, 3)
# plotData = phi(0,3)
# ex1.plot_points(plotData, target, 4)
# plotData = phi(1,0)
# ex1.plot_points(plotData, target, 5)
# plotData = phi(1,1)
# ex1.plot_points(plotData, target, 6)
# plotData = phi(1,2)
# ex1.plot_points(plotData, target, 7)
# plotData = phi(2,0)
# ex1.plot_points(plotData, target, 8)
# plotData = phi(2,1)
# ex1.plot_points(plotData, target, 9)
# plotData = phi(3,0)
# ex1.plot_points(plotData, target, 10)


plt.show()