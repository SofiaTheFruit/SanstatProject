from mpl_toolkits import mplot3d
import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from math import pi
from scipy.spatial.distance import cdist



universeX = np.linspace(-1, 1, 41)
sigma = 0.3

trainX_list = []
testX_list = []
ttrain = []
ttest = []
Xext = []
Xext2 = []

for x1 in universeX:
    for x2 in universeX:
        t_true = x1**2 * 2.5 + (-0.5)*x2**3
        
        # Villkor för testdata
        if abs(x1) > 0.3 or abs(x2) > 0.3:
            testX_list.append([x1, x2])
            ttest.append(t_true + np.random.normal(0, sigma) + 0.25*np.random.normal(0, sigma))
            Xext2.append([1, x1**2, x2**3])
        # Villkor för träningsdata
        else:
            trainX_list.append([x1, x2])
            ttrain.append(t_true + np.random.normal(0, sigma))
            Xext.append([1, x1**2, x2**3])

Xext = np.array(Xext)
Xext2 = np.array(Xext2)
ttrain = np.array(ttrain)
ttest = np.array(ttest)

wML = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Xext), Xext)), np.transpose(Xext)), ttrain)

print("wML:")
print(wML)

betaML = 0
for i in range(len(trainX_list)):
    x1, x2 = trainX_list[i]
    betaML += np.square(ttrain[i] - np.dot(wML, np.transpose([1, x1**2, x2**3])))

betaML = betaML/len(ttrain)
betaML = 1/betaML

print("\nbetaML:")
print(betaML)

predT = []
for i in range(len(testX_list)):
    x1, x2 = testX_list[i]
    predT += [wML[0] + wML[1]*(x1**2) + wML[2]*(x2**3)]

total = 0
for i in range(len(predT)):
    total += np.square(predT[i] - ttest[i])
total = total/len(predT)

print("\nMSE:")
print(total)

test_x1 = np.array([pt[0] for pt in testX_list])
test_x2 = np.array([pt[1] for pt in testX_list])
train_x1 = np.array([pt[0] for pt in trainX_list])
train_x2 = np.array([pt[1] for pt in trainX_list])

alpha = [0.2,0.8,2]
beta = 1/(sigma**2)

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

print("\nBayesian MSE:")
for i in range(len(alpha)):
    mse_bayesian = 0
    for j in range(len(ttest)):
        mse_bayesian += np.square(my[i][j] - ttest[j])
    mse_bayesian = mse_bayesian / len(ttest)
    print(f"Alpha {alpha[i]}: {mse_bayesian}")


plt.figure(1)
ax = plt.axes(projection = '3d')
ax.set_title('Test Data')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('ttest')
ax.plot3D(test_x1, test_x2, ttest, 'o')

plt.figure(2)
ax = plt.axes(projection = '3d')
ax.set_title('Training Data')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('ttrain')
ax.plot3D(train_x1, train_x2, ttrain, 'o')

fig = plt.figure(3, figsize=(15, 5))

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.set_title('Alpha: 0.2')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('Predicted T')
ax.plot_trisurf(test_x1, test_x2, my[0], color='pink')

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.set_title('Alpha: 0.8')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('Predicted T')
ax.plot_trisurf(test_x1, test_x2, my[1], color='pink')

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.set_title('Alpha: 2.0')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('Predicted T')
ax.plot_trisurf(test_x1, test_x2, my[2], color='pink')

plt.show()