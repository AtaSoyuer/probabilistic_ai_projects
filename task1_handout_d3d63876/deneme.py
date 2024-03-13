import os
import typing

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import make_scorer

from sklearn.kernel_approximation import Nystroem


posteriors = np.loadtxt('posterior.csv', delimiter=',', skiprows=1)
plt.figure()
plt.plot(posteriors[0,:])
plt.savefig('posterior.png')