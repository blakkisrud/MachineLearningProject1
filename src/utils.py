"""

Here we put functions that are used across the project.

The can be imported as:

from utils import function_name

or 

import utils as project_utils

"""

# Basic import statements - can be reduced

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import sys
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import os
from sklearn import preprocessing
import seaborn as sns
from sklearn.utils import resample

sns.set_style("whitegrid")

# Functions

def FrankeFunction(x,y, add_noise = False, sigma = 0.1):

    """
    Franke function, used for regression analysis.

    Args:
        x (float): x value
        y (float): y value
        add_noise (bool): Whether to add noise to the function
        sigma (float): Standard deviation of the noise
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    if add_noise:
        noise = np.random.normal(0, sigma, size=term1.shape)
        return term1 + term2 + term3 + term4 + noise
    else:
        return term1 + term2 + term3 + term4
    

