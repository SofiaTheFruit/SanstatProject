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

sigma = 0.3

t = []
for x1 in x[0]:
    temp = []
    for x2 in x[1]:
        temp += [x1*x1*2.5 + (-0.5)*x2*x2*x2 + np.random.normal(0,sigma)]
    t += [temp]

X,Y = np.meshgrid(x[0],x[1])

plt.figure(1)
ax = plt.axes(projection = '3d')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('t')
ax.plot3D(X,Y,np.array(t), 'o')
plt.show()