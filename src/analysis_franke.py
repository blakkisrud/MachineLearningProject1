"""

Code to run the analysis on the Frankes function.

If functios are re-used consider moving them to the utils.py file.

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
import utils as project_utils
import pandas as pd

# Global plotting parameters

sns.set_style("whitegrid")
sns.set_context("paper")

# Global variables

SIGMA = 0.1  # Noise level
OUTPUT_DIR = "output_tmp"  # Output directory for figures
STEP_SIZE = 0.01  # Step size for sampling the Franke function

# Running flags

DO_PART_A = False
DO_PART_B = False
DO_PART_C = True
DO_PART_D = False
DO_PART_E = False
DO_PART_F = False

# Helper functions

# Functions for doing the analysis, one function per subtask


def part_a():

    x = np.arange(0, 1, STEP_SIZE)
    y = np.arange(0, 1, STEP_SIZE)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x, y, franke_z, test_size=0.3)

    print("Size of the test and training data sets:")
    print(x_test.shape, x_train.shape, y_test.shape,
          y_train.shape, z_test.shape, z_train.shape)

    # Set up pandas frame for keeping scare of the results

    results = pd.DataFrame(
        columns=["Polynomial", "MSE_train", "MSE_test", "R2_train", "R2_test"])

    for p in range(1, 6):
        print("Polynomial degree: ", p)

        X_train = project_utils.generate_design_matrix(x_train, y_train, p)
        X_test = project_utils.generate_design_matrix(x_test, y_test, p)

        print("Size of X-matrix for training and testing:")
        print(X_train.shape, X_test.shape)

        MSE_train, R2_train, z_tilde_train, beta_ols = project_utils.OLS(
            X_train, z_train)

        z_tilde_test = X_test.dot(beta_ols)

        MSE_test = project_utils.MSE(z_test, z_tilde_test)
        R2_test = project_utils.R2(z_test, z_tilde_test)

        result = {"Polynomial": p, "MSE_train": MSE_train,
                  "MSE_test": MSE_test, "R2_train": R2_train, "R2_test": R2_test}

        results = results._append(result, ignore_index=True)

    # Plot the results

    project_utils.plot_mse_and_r2(results, 
                                  OUTPUT_DIR, 
                                  "franke_OLS_mse_r2.png",
                                  "OLS")


def part_b():

    x = np.arange(0, 1, STEP_SIZE)
    y = np.arange(0, 1, STEP_SIZE)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x, y, franke_z, test_size=0.3)

    print("Size of the test and training data sets:")

    print(x_test.shape, x_train.shape, y_test.shape,
          y_train.shape, z_test.shape, z_train.shape)

    # Set up pandas frame for keeping scare of the results

    results = pd.DataFrame(
        columns=["Polynomial", "MSE_train", "MSE_test", "R2_train", "R2_test", "Lambda"])

    for p in range(1, 12):

        print("Polynomial degree: ", p)

        X_train = project_utils.generate_design_matrix(x_train, y_train, p)
        X_test = project_utils.generate_design_matrix(x_test, y_test, p)

        print("Size of X-matrix for training and testing:")
        print(X_train.shape, X_test.shape)

        for lmb in np.logspace(-8, -3, 20):

            MSE_train, R2_train, z_tilde_train, beta_ols = project_utils.ridge(
                X_train, z_train, lmb)

            z_tilde_test = X_test.dot(beta_ols)

            MSE_test = project_utils.MSE(z_test, z_tilde_test)
            R2_test = project_utils.R2(z_test, z_tilde_test)

            result = {"Polynomial": p, "MSE_train": MSE_train,
                      "MSE_test": MSE_test, "R2_train": R2_train, "R2_test": R2_test,
                      "Lambda": lmb}

            results = results._append(result, ignore_index=True)

    print(results)

    # Plot the results

    project_utils.plot_mse_and_r2(results, 
                                  OUTPUT_DIR, 
                                  "franke_ridge_mse_r2.png",
                                  "Ridge")

def part_c():

    x = np.arange(0, 1, STEP_SIZE)
    y = np.arange(0, 1, STEP_SIZE)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x, y, franke_z, test_size=0.3)
    
    results = pd.DataFrame(
        columns=["Polynomial", "MSE_train", "MSE_test", "R2_train", "R2_test", "Lambda"])
    
    for p in range(1,6):

        X_train = project_utils.generate_design_matrix(x_train, y_train, p)
        X_test = project_utils.generate_design_matrix(x_test, y_test, p)

        for lmb in np.logspace(-8, -3, 12):
            print(lmb)

            clf = linear_model.Lasso(alpha=lmb, 
                                     fit_intercept=False, 
                                     max_iter=10000)

            clf.fit(X_train, z_train)

            z_pred_train = clf.predict(X_train)
            z_pred_test = clf.predict(X_test)

            MSE_train = mean_squared_error(z_train, z_pred_train)
            MSE_test = mean_squared_error(z_test, z_pred_test)

            R2_train = r2_score(z_train, z_pred_train)
            R2_test = r2_score(z_test, z_pred_test)

            result = {"Polynomial": p, "MSE_train": MSE_train,
                      "MSE_test": MSE_test, "R2_train": R2_train, "R2_test": R2_test,
                      "Lambda": lmb}
            
            results = results._append(result, ignore_index=True)

    print(results)

    # Plot the results

    project_utils.plot_mse_and_r2(results,
                                    OUTPUT_DIR,
                                    "franke_lasso_mse_r2.png",
                                    "Lasso")

        



# Run the parts


if DO_PART_A:
    part_a()
if DO_PART_B:
    part_b()
if DO_PART_C:
    part_c()
