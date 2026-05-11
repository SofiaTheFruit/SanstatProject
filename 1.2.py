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
    
    trainingSubsetsT += temp

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

subsetLikelyhood = [np.zeros(W0arr.shape),np.zeros(W0arr.shape),np.zeros(W0arr.shape),np.zeros(W0arr.shape)]

for l in range(len(trainingSubsetsX)):
    print("Fixing subset: " + str(l + 1))
    for i in range(W0arr.shape[0]):
        for j in range(W0arr.shape[1]):
            # Grabs positional values for w0 and w1
            w0 = W0arr[i,j]
            w1 = W1arr[i,j]

            temp = 1
            for k in range(len(trainingSubsetsX[l])):
                temp = temp * norm.pdf(trainingSubsetsT[l], w0 + w1*trainingSubsetsX[l][k], np.sqrt(1/beta)) # Perform the Gaussian product equation as specified in eq 17

            subsetLikelyhood[l][i,j] = temp # Sets final calculated value in the likelyhood plot

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
plt.show()
