import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from math import pi
from scipy.spatial.distance import cdist


# To draw n samples from multivariate Gaussian distribution with mu and Cov :
# f = np . random . multivariate_normal ( mu , Cov , n )

w0list = np.linspace (-3.0, 3.0, 200)
w1list = np.linspace (-3.0, 3.0, 200)
W0arr, W1arr = np.meshgrid (w0list, w1list)
pos = np.dstack ((W0arr, W1arr))

trainingX = np.arange(-1, 1.01, 0.01)
trainingT = []
for x in trainingX:
    trainingT = trainingT + [-1.2 + 0.9*x + np.random.normal(0,0.2)]

ones = []
for x in trainingX:
    ones = ones + [1]

Xext = [ones, trainingX]

#wML = np.dot(np.dot(np.linalg.inv(np.transpose(np.dot(Xext,np.transpose(Xext)))), Xext), np.transpose(trainingT))

alpha = 2
beta = 2
variance = 1 / alpha


# set your mu vector and Cov array
mu = [0, 0]
Cov = [[variance, 0.0],[0.0, variance]] # Was we do in reality is to multiply the variance by the identity matrix

rv = multivariate_normal(mu, Cov)
Wpriorpdf = rv.pdf(pos)


"""plt.figure(1)
plt.contour (W0arr, W1arr, Wpriorpdf)
plt.show()"""

likelyhood = np.zeros(W0arr.shape) # Creates all the available points
print("Training")
for i in range(W0arr.shape[0]):
    for j in range(W0arr.shape[1]):
        # Grabs positional values for w0 and w1
        w0 = W0arr[i,j]
        w1 = W1arr[i,j]

        temp = 1
        for k in range(len(trainingX)):
            temp = temp * norm.pdf(trainingT[k], w0 + w1*trainingX[k], np.sqrt(1/beta)) # Perform the Gaussian product equation as specified in eq 17
            
        likelyhood[i,j] = temp # Sets final calculated value in the likelyhood plot
print("Finished training")
"""plt.figure(2)
plt.contour(W0arr, W1arr, likelyhood)
plt.show()"""

sInv = np.array([[alpha,0],[0,alpha]]) + beta*np.dot(Xext, np.transpose(Xext))
mN = beta * np.dot(np.dot(np.linalg.inv(sInv),Xext), np.transpose(trainingT))

posterior = multivariate_normal(mN, np.linalg.inv(sInv))
Wposteriorpdf = posterior.pdf(pos)

"""plt.figure(3)
plt.contour(W0arr, W1arr, Wposteriorpdf)
plt.show()"""

samples = posterior.rvs(5)

x = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, 1.1, 1.2, 1.3, 1.4, 1.5])
y = [0,0,0,0,0]

testingT = []
for point in x:
    testingT += [-1.2 + 0.9*point + np.random.normal(0,0.2)]

for i in range(5):
    y[i] = samples[i][0] + x*samples[i][1]


plt.figure(4)
for line in y:
    plt.plot(x,line)
plt.plot(trainingX, trainingT, 'o')
plt.plot(x, testingT, 'o')
plt.show()
