import numpy as np
import matplotlib.pyplot as plt
import linreg_and_perceptron as ex1

smth = np.load('ex2_data0.npz')
print smth.files

data = smth['x']
target = smth['y']

# plot data on figure 1
ex1.plot_points(data, target, 1)

# transforming the data by squaring all entries
# phi(x1, x2) = (x1^2, x2^2)
# accomplished by multiplying the data matrice with itself
sqrData = data * data

plt.figure(2)
plt.xlim(0,1)
plt.ylim(0,1)
ex1.plot_points(sqrData, target, 2)

# add column of 1's to sqrData for perceptron algorithm
sqrData = np.c_[ np.ones(sqrData.shape[0]), sqrData ] 
ex1.plot_perceptron(sqrData, target, 0, 1, 2)

# plot squared data on figure 2
xs = ys = np.linspace(-1, 1, 100)
y, x = np.meshgrid(ys, xs)
# shape of x,y is (100,100) now

# reshape into vector of length 10000 (100x100)
xVector = np.asarray(x).reshape(-1)
yVector = np.asarray(y).reshape(-1)
oneVector = np.ones(xVector.shape[0])

# squaring stuff
xVector = xVector*xVector
yVector = yVector*yVector

# combining to a (10000, 3)-shaped matrix
resultCoordMatrix = np.c_[oneVector, xVector, yVector]

# multiplying by weight-vector from perceptron on data points
w = ex1.perceptron(sqrData, target, [0,0,0])
resultCoordMatrix = np.dot(w, np.transpose(resultCoordMatrix))

# reshape our (10000, 3)-matrix back to the (100,100) needed by contour
z = resultCoordMatrix.reshape(100,100)


plt.figure(1)
plt.contour(xs, ys, z, [0])


plt.show()
