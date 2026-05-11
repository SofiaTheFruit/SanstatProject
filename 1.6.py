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

wML = np.dot(np.dot(np.linalg.inv(np.transpose(np.dot(Xext,np.transpose(Xext)))), Xext), np.transpose(trainingT))

betaML = 0
for i in range(len(trainingT)):
    betaML = betaML + np.square(trainingT[i] - np.dot(wML,np.transpose([1, trainingX[i]])))

betaML = betaML/len(trainingT)
betaML = 1/betaML

print("Beta_ML:")
print(betaML)

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
"""
plt.figure(3)
plt.contour(W0arr, W1arr, Wposteriorpdf)
plt.show()

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
plt.show()"""

testingX = [-1.5, -1.4, -1.3, -1.2, -1.1, 1.1, 1.2, 1.3, 1.4, 1.5]

#np.dot(mN[0], [1, test])
#np.sqrt(1/beta + np.dot(np.dot([1, test],np.linalg.inv(sInv[0])), np.transpose([1, test])))

my = []
stdDiv = []
mlPred = []
for i in range(len(testingX)):
    my = my + [np.dot(mN, [1, testingX[i]])]

    stdDiv = stdDiv + [np.sqrt(1/beta + np.dot(np.dot([1, testingX[i]],np.linalg.inv(sInv)), np.transpose([1, testingX[i]])))]

    mlPred = mlPred + [wML[0] + wML[1]*testingX[i]]


plt.figure(6)
plt.errorbar(testingX, my, stdDiv)
plt.plot(testingX, mlPred)
plt.show()