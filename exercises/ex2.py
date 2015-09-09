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

# plot squared data on figure 2
ex1.plot_points(sqrData, target, 2)

# add column of 1's to sqrData for perceptron algorithm
sqrData = np.c_[ np.ones(sqrData.shape[0]), sqrData ] 

# plot perceptron line
ex1.plot_perceptron(sqrData, target, 0, 1, 2)


plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
