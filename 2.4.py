from mpl_toolkits import mplot3d
import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from math import pi
from scipy.spatial.distance import cdist



universeX = np.linspace(-1, 1, 41)

x = [[], []]
for i in range(2):
    usedIndex = []
    for j in range(10):
        while(1):
            index = np.random.randint(0, len(universeX))
            if index not in usedIndex:
                x[i] += [float(universeX[index])]
                usedIndex += [index]
                break
    x[i].sort()

testX = [[], []]
trainX = [[], []]
for i in range(10):
    if abs(x[0][i]) > 0.3:
        testX[0] += [x[0][i]]
    else:
        trainX[0] += [x[0][i]]

    if abs(x[1][i]) > 0.3:
        testX[1] += [x[1][i]]
    else:
        trainX[1] += [x[1][i]]

sigma = 0.3

ttest = []
for x1 in testX[0]:
    temp = []
    for x2 in testX[1]:
        temp += [x1*x1*2.5 + (-0.5)*x2*x2*x2 + np.random.normal(0,sigma) + 0.25*np.random.normal(0,sigma)]
    ttest += temp

teX,teY = np.meshgrid(testX[0],testX[1])

ttrain = []
for x1 in trainX[0]:
    temp = []
    for x2 in trainX[1]:
        temp += [x1*x1*2.5 + (-0.5)*x2*x2*x2 + np.random.normal(0,sigma)]
    ttrain += temp

trX,trY = np.meshgrid(trainX[0],trainX[1])

Xext = []
for x1 in trainX[0]:
    for x2 in trainX[1]:
        Xext += [[1, x1, x2]]

Xext2 = []
for x1 in testX[0]:
    for x2 in testX[1]:
        Xext2 += [[1, x1, x2]]

wML = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Xext), Xext)), np.transpose(Xext)), ttrain)

print("wML:")
print(wML)

betaML = 0
for i in range(len(trainX[0])):
    for j in range(len(trainX[1])):
        betaML = betaML + np.square(ttrain[i+len(trainX[0])*j] - np.dot(wML,np.transpose([1, trainX[0][i], trainX[1][j]])))

betaML = betaML/len(ttrain)
betaML = 1/betaML

print("betaML:")
print(betaML)

predT = []
for i in range(len(testX[0])):
    for j in range(len(testX[1])):
        predT += [wML[0] + wML[1]*testX[0][i] + wML[2]*testX[1][j]*testX[1][j]]

total = 0
for i in range(len(predT)):
    total += np.square(predT[i] - ttest[i])
total = total/len(predT)

print("MSE:")
print(total)

alpha = 0.2
beta = 1/sigma



sInv = np.array([[alpha,0, 0],[0,alpha, 0], [0, 0, alpha]]) + beta*np.dot(np.transpose(Xext), Xext)
mN = beta * np.dot(np.dot(np.linalg.inv(sInv),np.transpose(Xext)), np.transpose(ttrain))
print(mN)

my = []
stdDiv = []
for x in Xext2:
    my = my + [np.dot(mN, x)]

    stdDiv = stdDiv + [np.sqrt(1/beta + np.dot(np.dot(x,np.linalg.inv(sInv)), np.transpose(x)))]

print("my")
print(my)







plt.figure(1)
ax = plt.axes(projection = '3d')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('ttest')
ax.plot3D(teX.flatten(), teY.flatten(), np.array(ttest).flatten(), 'o')
plt.show()

plt.figure(2)
ax = plt.axes(projection = '3d')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('ttrain')
ax.plot3D(trX.flatten(), trY.flatten(), np.array(ttrain).flatten(), 'o')
plt.show()


plt.figure(1)
ax = plt.axes(projection = '3d')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('ttest')
ax.plot_surface(teX, teY, np.array(my).reshape(teX.shape), color='pink')
ax.errorbar(teX.flatten(), teY.flatten(), my, zerr=stdDiv, fmt='^')

plt.show()
