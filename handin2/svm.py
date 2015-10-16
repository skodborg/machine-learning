import scipy as sc
import numpy as np
import load_data
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.externals import joblib

t_t_imgs = None
t_v_imgs = None

def run(svc, n_comp):
	t_imgs, t_lbls, v_imgs, v_lbls = load_data.auDigit_data()

	pca = PCA(n_components=n_comp)
	pca.fit(t_imgs)

	# export pca model
	joblib.dump(pca, 'au_trained_pca.pkl')

	t_t_imgs = pca.transform(t_imgs)
	t_v_imgs = pca.transform(v_imgs)

	clf = svc
	clf.fit(t_t_imgs, t_lbls)

	# export trained model
	joblib.dump(clf, 'au_trained_svm.pkl')
	
	# in-sample
	predictions = clf.predict(t_t_imgs)
	hits = np.sum((t_lbls == predictions))
	hits_pct = float(hits)/t_t_imgs.shape[0] * 100
	print("in: " + str(hits_pct))

	# out-of-sample
	predictions = clf.predict(t_v_imgs)
	hits = np.sum((v_lbls == predictions))
	hits_pct = float(hits)/t_v_imgs.shape[0] * 100
	print("out: " + str(hits_pct))


def test(n_comp):
	# C = [-10.0, -7.0, -5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

	# Linear Kernel
	# print("Linear Kernel")
	# for i in C:
	# 	c = 3**i
	# 	print()
	# 	print("TRYING WITH C = " + str(3) + "**" + str(i))
	# 	run(SVC(kernel='linear', C=c))

	# degrees = [2, 3, 4]
	# # Polynomial Kernel
	# print("Polynomial Kernel")
	# for i in C:
	# 	for d in degrees:
	# 		c = 3**i
	# 		print()
	# 		print("TRYING WITH C = " + str(3) + "**" + str(i) + " AND d = " + str(d))
	# 		run(SVC(kernel='poly', C=c, degree=d))

	# gammas = [-10.0, -5.0, -1.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0]
	# RBF Kernel
	# print("RBF Kernel")
	# for i in C:
	# 	for g in gammas:
	# 		gamma = 3**g
	# 		c = 3**i
	# 		print()
	# 		print("TRYING WITH C = " + str(3) + "**" + str(i) + " AND g = " + str(3) + "**" + str(g))
	# 		run(SVC(kernel='rbf', C=c, gamma=gamma), n_comp)

	# Best found configuration used below
	C = [6.0]
	gammas = [-5.0]
	# RBF Kernel
	print("Training using RBF Kernel with best found parameters for the auDigits data set")
	for i in C:
		for g in gammas:
			gamma = 3**g
			c = 3**i
			run(SVC(kernel='rbf', C=c, gamma=gamma), n_comp)

if __name__ == '__main__':
	print("n_comp = 200")
	# for i in range(0,20):
	test(200)









