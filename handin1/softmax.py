import numpy as np
import matplotlib.pyplot as plt
import math
import time

autraindatafile = np.load('auTrainMerged.npz')

autrainlabels = autraindatafile['labels']
autrainimages = autraindatafile['digits']

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

def soft_cost(X, Y, theta):
	fn_cost = lambda X, Y, theta: -(np.sum(Y * np.log(matrix_softmax(np.dot(X, theta)))))
	gradient = np.dot(-X.T, (Y - matrix_softmax(np.dot(X, theta))))
	return fn_cost, gradient

def soft_run(X, Y, theta, useRegularization=False, lambda_i=0):
	w = theta
	my = 0.05
	current_cost = 100000000.0
	costImprovLimit = 0.001
	costImprovement = 1.0
	iters = []
	costs = []
	t = 0
	# interactive plotting as we progress
	plt.ion()
	plt.show()
	# for t in range(0, 200):
	while costImprovement > costImprovLimit:
		costfn, gradient = soft_cost(X, Y, w)
		gradient = (1.0 / X.shape[0]) * gradient
		old_cost = current_cost
		current_cost = (1.0 / X.shape[0]) * costfn(X, Y, w)
		# plot cost per iteration
		costs.append(current_cost)
		iters.append(t)

		if (old_cost < current_cost):
			print "OUCH! increased cost, returning current weight. Latest cost impr. of " + str(costImprovement) + " using step size " + str(my)
			print "cost at: " + str(current_cost)
			# plot costs per iteration
			# if (current_cost > 0.15):
				# plt.plot(iters, costs, 'r-')
				# plt.show()
			return w
		costImprovement = ((old_cost - current_cost) / old_cost) * 100.0
		
		# print str(t) + ": " + str(current_cost)

		v = -(gradient)
		w = w + my * v

		# interactive plotting as we progress
		plt.plot(iters, costs, 'r-')
		plt.draw()
		t += 1
	return w


# someX = np.concatenate([singleX, singleX, singleX]).reshape(3,9)
# someY = np.array([0,1,0,1,1,0]).reshape(3,2)
# someTheta = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]).reshape(9,2)

# print "X: " + str(someX.shape)
# print "Y: " + str(someY.shape)
# print "Theta: " + str(someTheta.shape)

# print np.dot(someY[3], np.log(softmax(np.dot(someTheta.T, someX[3]))))
# print np.dot(-someX.T, someY - matrix_softmax(np.dot(someX, someTheta)))

# myXs = append_ones_column_first(autrainimages)
# myYs = convert_labels(autrainlabels)
# myTheta = np.zeros(myXs.shape[1]*myYs.shape[1]).reshape(myXs.shape[1], myYs.shape[1])
# soft_run(myXs, myYs, myTheta)







def classifyDigits(X, Y, theta):
	# initX = append_ones_column_first(images)
	# initY = convert_labels(labels)	
	# initTheta = np.zeros(785*10).reshape(785,10)
	# initTheta[:,0] = 1

	print "running soft_run"
	start = time.clock()
	learnedThetas = soft_run(X, Y, theta)
	end = time.clock()
	print "finished in " + str(end-start) + " seconds"
	np.savez("softmax_matrixOps_learnedThetas_st1_cl01.npz", theta=learnedThetas)

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

trainingimgs, validationimgs, traininglabels, validationlabels = training_validation_split(autrainimages, autrainlabels, 5)

initY = convert_labels(traininglabels)
initX = append_ones_column_first(trainingimgs)
initTheta = np.zeros(785*10).reshape(785,10)
initTheta[:,0] = 1

classifyDigits(initX, initY, initTheta)



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





myThetas = np.load("softmax_matrixOps_learnedThetas_st1_cl01.npz")['theta']
compare(0, validationimgs.shape[0], myThetas, validationimgs, validationlabels)
