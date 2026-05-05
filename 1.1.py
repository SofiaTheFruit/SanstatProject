import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import pi
from scipy.spatial.distance import cdist


# To draw n samples from multivariate Gaussian distribution with mu and Cov :
# f = np . random . multivariate_normal ( mu , Cov , n )

w0list = np.linspace (-3.0, 3.0, 200)
w1list = np.linspace (-3.0, 3.0, 200)
W0arr, W1arr = np.meshgrid (w0list, w1list)
pos = np.dstack ((W0arr, W1arr))

alpha = 2
variance = 1 / alpha


# set your mu vector and Cov array
mu = [0, 0]
Cov = [[variance, 0.0],[0.0, variance]] # Was we do in reality is to multiply the variance by the identity matrix

rv = multivariate_normal(mu, Cov)
Wpriorpdf = rv.pdf(pos)

plt.figure(1)
plt.contour (W0arr, W1arr, Wpriorpdf)
plt.show()