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

trainingSubsetOne = np.array([-1, 0.35, 0.79])
trainingSubsetTwo = np.array([-1, -0.89, -0.65, -0.55, -0.43, -0.03, 0, 0.35, 0.5, 0.79])
trainingSubsetThree = np.array([-1, -0.89, -0.79, -0.65, -0.55, -0.5, -0.43, -0.35 -0.03, 0, 0.01, 0.03, 0.35, 0.43, 0.5, 0.55, 0.65, 0.79, 0.89, 1])
trainingSubsetFour = []
for i in range(100):
    trainingSubsetFour += [float(trainingX[np.random.randint(0,len(trainingX))])]

trainingSubsetFour.sort()
trainingSubsetFour = np.array(trainingSubsetFour)

trainingSubsetsX = [trainingSubsetOne, trainingSubsetTwo, trainingSubsetThree, trainingSubsetFour]

trainingSubsetsT = []
for i in range(len(trainingSubsetsX)):
    temp =[]
    for x in trainingSubsetsX[i]:
        temp += [-1.2 + 0.9*x + np.random.normal(0,0.2)]
    trainingSubsetsT += [temp]

trainingT = []
for x in trainingX:
    trainingT = trainingT + [-1.2 + 0.9*x + np.random.normal(0,0.2)]

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


plt.figure(1)
plt.contour (W0arr, W1arr, Wpriorpdf)

subsetLikelyhood = [1,1,1,1]

for l in range(len(trainingSubsetsX)):
    print("Fixing subset: " + str(l + 1))
    for k in range(len(trainingSubsetsX[l])):
        subsetLikelyhood[l] = subsetLikelyhood[l] * norm.pdf(trainingSubsetsT[l][k], W0arr + W1arr*trainingSubsetsX[l][k], np.sqrt(1/beta)) # Perform the Gaussian product equation as specified in eq 17


    """for i in range(W0arr.shape[0]):
        for j in range(W0arr.shape[1]):
            # Grabs positional values for w0 and w1
            w0 = W0arr[i,j]
            w1 = W1arr[i,j]

            temp = 1
            for k in range(len(trainingSubsetsX[l])):
                temp = temp * norm.pdf(trainingSubsetsT[l][k], w0 + w1*trainingSubsetsX[l][k], np.sqrt(1/beta)) # Perform the Gaussian product equation as specified in eq 17

            subsetLikelyhood[l][i,j] = temp # Sets final calculated value in the likelyhood plot"""

print("Training")
likelyhood = 1
for k in range(len(trainingX)):
    likelyhood = likelyhood * norm.pdf(trainingT[k], W0arr + W1arr*trainingX[k], np.sqrt(1/beta)) # Perform the Gaussian product equation as specified in eq 17


print("Finished training")
plt.figure(2)
plt.title("200 data points")
plt.contour(W0arr, W1arr, likelyhood)

plt.figure(3)
plt.subplot(2,2,1)
plt.title("3 data points")
plt.contour(W0arr, W1arr, subsetLikelyhood[0])
plt.subplot(2,2,2)
plt.title("10 data points")
plt.contour(W0arr, W1arr, subsetLikelyhood[1])
plt.subplot(2,2,3)
plt.title("20 data points")
plt.contour(W0arr, W1arr, subsetLikelyhood[2])
plt.subplot(2,2,4)
plt.title("100 data points")
plt.contour(W0arr, W1arr, subsetLikelyhood[3])


sInv = np.array([[alpha,0],[0,alpha]]) + beta*np.dot(Xext, np.transpose(Xext))
mN = beta * np.dot(np.dot(np.linalg.inv(sInv),Xext), np.transpose(trainingT))

posterior = multivariate_normal(mN, np.linalg.inv(sInv))
Wposteriorpdf = posterior.pdf(pos)

plt.figure(4)
plt.contour(W0arr, W1arr, Wposteriorpdf)

samples = posterior.rvs(5)

x = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, 1.1, 1.2, 1.3, 1.4, 1.5])
y = [0,0,0,0,0]

testingT = []
for point in x:
    testingT += [-1.2 + 0.9*point + np.random.normal(0,0.2)]

for i in range(5):
    y[i] = samples[i][0] + x*samples[i][1]


plt.figure(5)
for line in y:
    plt.plot(x,line)
plt.plot(trainingX, trainingT, 'o')
plt.plot(x, testingT, 'o')

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