from mpl_toolkits import mplot3d
import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from math import pi
from scipy.spatial.distance import cdist
import random


universeX = np.linspace(-1, 1, 41)
sigma = 0.3
alpha_values = [0.2, 0.8, 2.0]
beta = 1 / (sigma**2)

trainX_list = []
testX_list = []
ttrain = []
ttest = []
Xext = []
Xext2 = []
full_trainX_list = []
full_ttrain = []

for x1 in universeX:
    for x2 in universeX:
        t_true = x1**2 * 2.5 + (-0.5)*x2**3
        if abs(x1) > 0.3 or abs(x2) > 0.3:
            testX_list.append([x1, x2])
            ttest.append(t_true + np.random.normal(0, sigma) + 0.25*np.random.normal(0, sigma))
            Xext2.append([1, x1**2, x2**3])
        else:
            full_trainX_list.append([x1, x2])
            full_ttrain.append(t_true + np.random.normal(0, sigma))

Xext2 = np.array(Xext2)
ttest = np.array(ttest)

train_fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]

ml_mse_history = []
bayes_mse_history = {alpha: [] for alpha in alpha_values}
sample_sizes = []

for fraction in train_fractions:
    num_samples = max(4, int(len(full_trainX_list) * fraction)) # Minst 4 punkter så ML inte kraschar helt
    sample_sizes.append(num_samples)
    
    indices = random.sample(range(len(full_trainX_list)), num_samples)
    
    trainX_sub = [full_trainX_list[i] for i in indices]
    ttrain_sub = np.array([full_ttrain[i] for i in indices])
    
    Xext_sub = []
    for pt in trainX_sub:
        Xext_sub.append([1, pt[0]**2, pt[1]**3])
    Xext_sub = np.array(Xext_sub)

    try:
        wML = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Xext_sub), Xext_sub)), np.transpose(Xext_sub)), ttrain_sub)
        
        predT = []
        for pt in testX_list:
            predT.append(wML[0] + wML[1]*(pt[0]**2) + wML[2]*(pt[1]**3))
            
        ml_mse = np.mean(np.square(np.array(predT) - ttest))
        ml_mse_history.append(ml_mse)
    except np.linalg.LinAlgError:
        ml_mse_history.append(np.nan)

    for alpha in alpha_values:
        sInv = np.array([[alpha, 0, 0], [0, alpha, 0], [0, 0, alpha]]) + beta * np.dot(np.transpose(Xext_sub), Xext_sub)
        mN = beta * np.dot(np.dot(np.linalg.inv(sInv), np.transpose(Xext_sub)), np.transpose(ttrain_sub))
        
        my = []
        for x in Xext2:
            my.append(np.dot(mN, x))
            
        bayes_mse = np.mean(np.square(np.array(my) - ttest))
        bayes_mse_history[alpha].append(bayes_mse)

plt.figure(figsize=(10, 6))

plt.plot(train_fractions, ml_mse_history, 'k--', linewidth=2, marker='X', label='Maximum Likelihood (ML)')

colors = ['b', 'g', 'r']
for idx, alpha in enumerate(alpha_values):
    plt.plot(train_fractions, bayes_mse_history[alpha], color=colors[idx], linewidth=2, marker='o', label=f'Bayesian (Alpha={alpha})')

plt.title('MSE Jämförelse: ML vs Bayesian vid krympande träningsdata')
plt.xlabel('mängd av dataset')
plt.ylabel('MSE')
plt.yscale('log') 
plt.gca().invert_xaxis() 
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()