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


def generate_design_matrix(x, y, n):
    """
    Here, x and y are the x and y values of the data points, respectively, and n is the polynomial degree.
    """

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X


def MSE(y, y_tilde):

    n = len(y)
    return np.sum((y - y_tilde)**2)/n


def R2(y, y_tilde):

    n = len(y)
    return 1 - np.sum((y - y_tilde)**2)/np.sum((y - np.mean(y))**2)


def OLS(X, z):

    """
    X: Design matrix
    z: Data

    Returns: MSE, R2, z_tilde, beta

    """

    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
    z_tilde = X.dot(beta)
    return MSE(z, z_tilde), R2(z, z_tilde), z_tilde, beta

def ridge(X, z, lmb):

    """
    X: Design matrix
    z: Data
    lmb: Ridge parameter

    Returns: MSE, R2, z_tilde, beta

    """

    beta = np.linalg.inv(X.T.dot(X) + lmb*np.eye(X.shape[1])).dot(X.T).dot(z)
    z_tilde = X.dot(beta)

    return MSE(z, z_tilde), R2(z, z_tilde), z_tilde, beta

def plot_mse_and_r2_OLS(result_frame, output_dir, filename):

    """
    For plotting the MSE and R2 as a function of the polynomial degree.

    For the OLS-analysis

    Arguments:
        result_frame (pandas.DataFrame): Contains the results from the regression analysis
        output_dir (str): Output directory
        filename (str): Filename of the plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)

    ax.plot(result_frame["Polynomial"], result_frame["MSE_train"], label="MSE train")
    ax.plot(result_frame["Polynomial"], result_frame["MSE_test"], label="MSE test")

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")

    ax.legend()

    ax = fig.add_subplot(1,2,2)

    ax.plot(result_frame["Polynomial"], result_frame["R2_train"], label="R2 train")
    ax.plot(result_frame["Polynomial"], result_frame["R2_test"], label="R2 test")

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("R2")

    ax.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, filename))

def plot_mse_and_r2(result_frame, output_dir, filename, type):

    """
    Plot of the ridge, LASSO or OLS regression analysis.
    """

    if type == "OLS":
        plot_mse_and_r2_OLS(result_frame, output_dir, filename)

    elif type == "Ridge" or "LASSO":

        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)

        sns.scatterplot(x = "Lambda",
                        y = "MSE_train",
                        data = result_frame,
                        hue = "Polynomial")
    
        ax.set_xlabel("Lambda")
        ax.set_ylabel("MSE")
        ax.set_xscale("log")
    
        ax = fig.add_subplot(2,2,2)
    
        sns.scatterplot(x = "Lambda",
                        y = "MSE_test",
                        data = result_frame,
                        hue = "Polynomial")

        ax.set_xlabel("Lambda")
        ax.set_ylabel("MSE")
        ax.set_xscale("log")

        ax.legend()

        ax = fig.add_subplot(2,2,3)

        sns.scatterplot(x = "Lambda",
                        y = "R2_train",
                        data = result_frame,
                        hue = "Polynomial")
    
        ax.set_xlabel("Lambda")
        ax.set_ylabel("R2")
        ax.set_xscale("log")

        ax = fig.add_subplot(2,2,4)

        sns.scatterplot(x = "Lambda",
                        y = "R2_test",
                        data = result_frame,
                        hue = "Polynomial")
    
        ax.set_xlabel("Lambda")
        ax.set_ylabel("R2")
        ax.set_xscale("log")

        ax.legend()

        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, filename))

def plot_bias_variance_tradeoff(result_frame, output_dir, filename):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(result_frame["Polynomial"], 
            result_frame["Bias"], 
            label="Bias")
    
    ax.plot(result_frame["Polynomial"],
            result_frame["Variance"],
            label="Variance")
    
    ax.plot(result_frame["Polynomial"],
            result_frame["Error"],
            label="Error")
    
    ax.set_xlabel("Polynomial degree")

    ax.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, filename))



def FrankeFunction(x, y, add_noise=False, sigma=0.1):
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
