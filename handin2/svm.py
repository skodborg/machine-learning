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
	C = [-10, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3]
	# Linear Kernel
	for i in C:
		c = 3**i
		print()
		print("TRYING WITH C = " + str(3) + "**" + str(i))
		run(SVC(kernel='linear', C=c))

if __name__ == '__main__':
	test()
