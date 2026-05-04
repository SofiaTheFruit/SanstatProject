import numpy as np
import matplotlib . pyplot as plt
from scipy . stats import multivariate_normal

w0list = np . linspace ( -3.0 , 1.0 , 200)
w1list = np . linspace ( -2.0 , 2.0 , 200)
W0arr , W1arr = np . meshgrid ( w0list , w1list )
pos = np . dstack (( W0arr , W1arr ))




# set your mu vector and Cov array

mu = [0.0, 0.0]
Cov = [[1.0, 0.5],[0.5, 1.0]]

rv = multivariate_normal ( mu , Cov )
Wpriorpdf = rv . pdf ( pos )
plt . contour ( W0arr , W1arr , Wpriorpdf )
plt . show ()