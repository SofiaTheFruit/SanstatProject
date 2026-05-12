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
sigma = 1.2
alpha = 2.0
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
sample_sizes = []

emp_var_train_hist = []
emp_var_test_hist = []
pred_var_train_hist = []
pred_var_test_hist = []

for fraction in train_fractions:
    num_samples = max(4, int(len(full_trainX_list) * fraction))
    sample_sizes.append(num_samples)
    
    indices = random.sample(range(len(full_trainX_list)), num_samples)
    
    trainX_sub = [full_trainX_list[i] for i in indices]
    ttrain_sub = np.array([full_ttrain[i] for i in indices])
    
    Xext_sub = []
    for pt in trainX_sub:
        Xext_sub.append([1, pt[0]**2, pt[1]**3])
    Xext_sub = np.array(Xext_sub)

    sInv = np.array([[alpha, 0, 0], [0, alpha, 0], [0, 0, alpha]]) + beta * np.dot(np.transpose(Xext_sub), Xext_sub)
    mN = beta * np.dot(np.dot(np.linalg.inv(sInv), np.transpose(Xext_sub)), np.transpose(ttrain_sub))
    
    my_test = []
    var_test = []
    for x in Xext2:
        my_test.append(np.dot(mN, x))
        var_test.append(1/beta + np.dot(np.dot(x, np.linalg.inv(sInv)), np.transpose(x)))
        
    my_train = []
    var_train = []
    for x in Xext_sub:
        my_train.append(np.dot(mN, x))
        var_train.append(1/beta + np.dot(np.dot(x, np.linalg.inv(sInv)), np.transpose(x)))

    emp_var_test_hist.append(np.var(np.array(my_test) - ttest))
    emp_var_train_hist.append(np.var(np.array(my_train) - ttrain_sub))
    
    pred_var_test_hist.append(np.mean(var_test))
    pred_var_train_hist.append(np.mean(var_train))


    print(f"{sample_sizes[0]:<20} | "f"{emp_var_test_hist[0]:<12.4f} | "f"{emp_var_train_hist[0]:<12.4f} | "f"{pred_var_test_hist[0]:<15.4f} | "f"{pred_var_train_hist[0]:<15.4f}")

plt.figure(figsize=(10, 6))

plt.plot(train_fractions, emp_var_test_hist, 'r-X', linewidth=2, label='MSE (Test)')
plt.plot(train_fractions, emp_var_train_hist, 'r--o', linewidth=2, alpha=0.6, label='MSE (Träning)')

plt.plot(train_fractions, pred_var_test_hist, 'b-X', linewidth=2, label='Prediktiv varians (Test)')
plt.plot(train_fractions, pred_var_train_hist, 'b--o', linewidth=2, alpha=0.6, label='Prediktiv varians (Träning)')

plt.title(f'(Alpha={alpha}, Sigma={sigma})')
plt.xlabel('andel av dataset')
plt.ylabel('Varians / MSE')
plt.gca().invert_xaxis() # Vänd x-axeln så vi krymper datan åt höger
plt.legend()
plt.grid(True, linestyle="--")
plt.show()