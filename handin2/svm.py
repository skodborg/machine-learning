import scipy as sc
import numpy as np
import load_data
from sklearn.svm import SVC

def run(svc):

	t_imgs, t_lbls, v_imgs, v_lbls = load_data.auDigit_data()

	clf = svc
	clf.fit(t_imgs, t_lbls)
	# in-sample
	predictions = clf.predict(t_imgs)
	hits = np.sum((t_lbls == predictions))
	print("in, hits: ", str(hits), "/", str(t_imgs.shape[0]), sep="")
	hits_pct = float(hits)/t_imgs.shape[0] * 100
	print("pct: " + str(hits_pct))
	# out-of-sample
	predictions = clf.predict(v_imgs)
	hits = np.sum((v_lbls == predictions))
	print("out, hits: ", str(hits), "/", str(v_imgs.shape[0]), sep="")
	hits_pct = float(hits)/v_imgs.shape[0] * 100
	print("pct: " + str(hits_pct))
	# print(test_images.shape[0])

def test():
	C = [-10.0, -7.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
	# Linear Kernel
	print("Linear Kernel")
	for i in C:
		c = 3**i
		print()
		print("TRYING WITH C = " + str(3) + "**" + str(i))
		run(SVC(kernel='linear', C=c))

	degrees = [2, 3, 4, 5]
	# Polynomial Kernel
	print("Polynomial Kernel")
	for i in C:
		for d in degrees:
			c = 3**i
			print()
			print("TRYING WITH C = " + str(3) + "**" + str(i) + " AND d = " + str(d))
			run(SVC(kernel='poly', C=c, degree=d))

	gammas = [-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0]
	# RBF Kernel
	print("RBF Kernel")
	for i in C:
		for g in gammas:
			gamma = 3**g
			c = 3**i
			print()
			print("TRYING WITH C = " + str(3) + "**" + str(i) + " AND g = " + str(3) + "**" + str(g))
			run(SVC(kernel='rbf', C=c, gamma=gamma))

if __name__ == '__main__':
	test()
