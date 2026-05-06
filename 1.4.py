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

trainingX = [[-1.5, -1.4, -1.3, -1.2, -1.1, 1.1, 1.2, 1.3, 1.4, 1.5],np.arange(-1, 1.01, 0.01)]
trainingT = [[],[]]
for i in range(len(trainingX)):
    for x in trainingX[i]:
        trainingT[i] = trainingT[i] + [-1.2 + 0.9*x + np.random.normal(0,0.2)]

ones = [[],[]]
for i in range(len(trainingX)):
    for x in trainingX[i]:
        ones[i] = ones[i] + [1]

Xext = [[ones[0], trainingX[0]],[ones[1],trainingX[1]]]

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

likelyhood = [np.zeros(W0arr.shape),np.zeros(W0arr.shape)] # Creates all the available points
for l in range(len(likelyhood)):
    print("Calculating training set: " + str(l+1))
    for i in range(W0arr.shape[0]):
        for j in range(W0arr.shape[1]):
            # Grabs positional values for w0 and w1
            w0 = W0arr[i,j]
            w1 = W1arr[i,j]

            temp = 1
            for k in range(len(trainingX[l])):
                temp = temp * norm.pdf(trainingT[l][k], w0 + w1*trainingX[l][k], np.sqrt(1/beta)) # Perform the Gaussian product equation as specified in eq 17
            
            likelyhood[l][i,j] = temp # Sets final calculated value in the likelyhood plot
print("Finished calculating")
"""plt.figure(2)
plt.contour(W0arr, W1arr, likelyhood)
plt.show()"""

sInv = [np.array([[alpha,0],[0,alpha]]) + beta*np.dot(Xext[0], np.transpose(Xext[0])),np.array([[alpha,0],[0,alpha]]) + beta*np.dot(Xext[1], np.transpose(Xext[1]))]
mN = [beta * np.dot(np.dot(np.linalg.inv(sInv[0]),Xext[0]), np.transpose(trainingT[0])),beta * np.dot(np.dot(np.linalg.inv(sInv[1]),Xext[1]), np.transpose(trainingT[1]))]

posterior = [multivariate_normal(mN[0], np.linalg.inv(sInv[0])), multivariate_normal(mN[1], np.linalg.inv(sInv[1]))]
Wposteriorpdf = [posterior[0].pdf(pos),posterior[1].pdf(pos)]
"""
plt.figure(3)
plt.contour(W0arr, W1arr, Wposteriorpdf)
plt.show()"""

samples = [posterior[0].rvs(5), posterior[1].rvs(5)]

x = [np.linspace(-1.5,1.5,30),np.arange(-1, 1.01, 0.01)]
y = [[0,0,0,0,0],[0,0,0,0,0]]

for i in range(5):
    y[0][i] = samples[0][i][0] + x[0]*samples[0][i][1]
    y[1][i] = samples[1][i][0] + x[1]*samples[1][i][1]


plt.figure(4)
for line in y[0]:
    plt.plot(x[0],line)
plt.plot(trainingX[0], trainingT[0], 'o')
plt.show()

plt.figure(5)
for line in y[1]:
    plt.plot(x[1],line)
plt.plot(trainingX[1], trainingT[1], 'o')
plt.show()