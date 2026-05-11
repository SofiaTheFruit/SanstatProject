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
print()

betaML = 0
for i in range(len(trainX[0])):
    for j in range(len(trainX[1])):
        betaML = betaML + np.square(ttrain[i+len(trainX[0])*j] - np.dot(wML,np.transpose([1, trainX[0][i], trainX[1][j]])))

betaML = betaML/len(ttrain)
betaML = 1/betaML

print("betaML:")
print(betaML)
print()

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
print()

alpha = [0.2,0.8,2]
beta = 1/sigma

my = [[],[],[]]
stdDiv = [[],[],[]]
sInv = [[],[],[]]
mN = [[],[],[]]
for i in range(len(alpha)):
    sInv[i] = np.array([[alpha[i],0, 0],[0,alpha[i], 0], [0, 0, alpha[i]]]) + beta*np.dot(np.transpose(Xext), Xext)
    mN[i] = beta * np.dot(np.dot(np.linalg.inv(sInv[i]),np.transpose(Xext)), np.transpose(ttrain))

    for x in Xext2:
        my[i] = my[i] + [np.dot(mN[i], x)]

        stdDiv[i] = stdDiv[i] + [np.sqrt(1/beta + np.dot(np.dot(x,np.linalg.inv(sInv[i])), np.transpose(x)))]







plt.figure(1)
ax = plt.axes(projection = '3d')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('ttest')
ax.plot3D(teX.flatten(), teY.flatten(), np.array(ttest).flatten(), 'o')

plt.figure(2)
ax = plt.axes(projection = '3d')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('ttrain')
ax.plot3D(trX.flatten(), trY.flatten(), np.array(ttrain).flatten(), 'o')


fig = plt.figure(3)
ax = fig.add_subplot(2,2,1,projection = '3d')
ax.set_title('Alpha: 0.2')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('Predicted T')
ax.plot_surface(teX, teY, np.array(my[0]).reshape(teX.shape), color='pink')
ax.errorbar(teX.flatten(), teY.flatten(), my[0], zerr=stdDiv[0], fmt='o')
ax = fig.add_subplot(2,2,2,projection = '3d')
ax.set_title('Alpha: 0.8')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('Predicted T')
ax.plot_surface(teX, teY, np.array(my[1]).reshape(teX.shape), color='pink')
ax.errorbar(teX.flatten(), teY.flatten(), my[1], zerr=stdDiv[1], fmt='o')
ax = fig.add_subplot(2,2,3,projection = '3d')
ax.set_title('Alpha: 2.0')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('Predicted T')
ax.plot_surface(teX, teY, np.array(my[2]).reshape(teX.shape), color='pink')
ax.errorbar(teX.flatten(), teY.flatten(), my[2], zerr=stdDiv[2], fmt='o')

plt.show()
