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

trainingX = [1.1, 1.3, 1.5]
trainingT = []
for x in trainingX:
    trainingT = trainingT + [-1.2 + 0.9*x]

ones = []
for x in trainingX:
    ones = ones + [1]

Xext = [ones, trainingX]

wML = np.dot(np.dot(np.linalg.inv(np.transpose(np.dot(Xext,np.transpose(Xext)))), Xext), np.transpose(trainingT))

alpha = 2
beta = 2
variance = 1 / alpha




# set your mu vector and Cov array
mu = [0, 0]
Cov = [[variance, 0.0],[0.0, variance]] # Was we do in reality is to multiply the variance by the identity matrix

rv = multivariate_normal(mu, Cov)
Wpriorpdf = rv.pdf(pos)

likelyhood = np.zeros(W0arr.shape)
for i in range(W0arr.shape[0]):
    for j in range(W0arr.shape[1]):
        w0 = W0arr[i,j]
        w1 = W1arr[i,j]

        error = trainingT - (np.transpose(Xext) @ np.array([w0,w1]))
        log_likelyhood = -0.5 * beta * np.sum(error**2)
        likelyhood[i,j] = np.exp(log_likelyhood)

#likelyhood = []
#for t in trainingT:
#    likelyhood = likelyhood * np.random.normal(t,1/beta)
plt.figure(1)
plt.contour (W0arr, W1arr, Wpriorpdf)
plt.show()

plt.figure(2)
plt.contour(W0arr, W1arr, likelyhood)
plt.show()