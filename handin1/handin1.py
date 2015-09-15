import numpy as np
import matplotlib.pyplot as plt
import his_ex2 as ex2
import math

datafile = np.load('mnistTrain.npz')

print datafile.files

labels = datafile['labels']
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


for i in range(0,10):
	plt.figure(i)
	visualize_image(i)
	print str(i) + str(labels[i])

# plt.show()