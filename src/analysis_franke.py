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
from sklearn.utils import resample

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
DO_PART_C = False
DO_PART_D = False
DO_PART_E = False
DO_PART_F = True

# Helper functions

# Functions for doing the analysis, one function per subtask

def part_a(plot_prediction=False):

    x = np.arange(0, 1, STEP_SIZE)
    y = np.arange(0, 1, STEP_SIZE)

    x, y = np.meshgrid(x, y)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)
    franke_z_flat = franke_z.ravel()

    x = np.ravel(x)
    y = np.ravel(y)

    index_vals = np.array(range(np.prod(franke_z.shape)))

    print(index_vals.shape)

    idx_train, idx_test = train_test_split(
        index_vals, test_size=0.3, random_state=42)

    # Save the train and test data sets as images

    project_utils.plot_train_test_image(
        franke_z, idx_test, idx_train, OUTPUT_DIR, "franke_training_test.png")

    results = pd.DataFrame(
        columns=["Polynomial", "MSE_test", "MSE_train", "R2_test", "R2_train"])

    for p in range(1, 12):

        X_train = project_utils.generate_design_matrix(
            x[idx_train], y[idx_train], p)
        X_test = project_utils.generate_design_matrix(
            x[idx_test], y[idx_test], p)

        MSE_train, R2_train, z_pred_train, beta = project_utils.OLS(
            X_train, franke_z_flat[idx_train])

        z_pred = X_test.dot(beta)
        z_test = franke_z_flat[idx_test]

        z_train = franke_z_flat[idx_train]

        MSE_test = project_utils.MSE(z_test, z_pred)

        R2_test = project_utils.R2(z_test, z_pred)

        result = {"Polynomial": p, "MSE_test": MSE_test,
                  "MSE_train": MSE_train, "R2_test": R2_test, "R2_train": R2_train}

        results = results._append(result, ignore_index=True)

        if plot_prediction:

            X_whole_image = project_utils.generate_design_matrix(x, y, p)

            Z_pred_whole_image = X_whole_image.dot(beta)

            Z_pred_whole_image = Z_pred_whole_image.reshape(franke_z.shape)

            img_predicton = np.zeros(index_vals.shape)

            img_predicton[idx_test] = z_pred

            img_predicton = img_predicton.reshape(franke_z.shape)

            # Plot the prediction

            fig, axes = plt.subplots(1, 3, figsize=(10, 5))

            fig.suptitle("Polynomial degree: " + str(p))

            axes[0].imshow(franke_z, cmap="viridis")
            axes[0].set_title("Franke function")

            axes[1].imshow(Z_pred_whole_image, cmap="viridis")
            axes[1].set_title("Prediction")

            axes[2].imshow(
                (franke_z - Z_pred_whole_image.reshape(franke_z.shape)), cmap="viridis")
            axes[2].set_title("Error")

            plt.savefig(os.path.join(
                OUTPUT_DIR, "franke_prediction_p" + str(p) + ".png"))

    print(results)

    project_utils.plot_mse_and_r2(results,
                                  OUTPUT_DIR,
                                  "franke_OLS_mse_r2.png",
                                  "OLS")

    sys.exit()

    img_train = np.zeros(index_vals.shape)
    img_test = np.zeros(index_vals.shape)

    img_train[idx_train] = franke_z_flat[idx_train]
    img_test[idx_test] = franke_z_flat[idx_test]

    img_train = img_train.reshape(franke_z.shape)
    img_test = img_test.reshape(franke_z.shape)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(franke_z, cmap="viridis")
    axes[0].set_title("Franke function")

    axes[1].imshow(img_train, cmap="viridis")
    axes[1].set_title("Train data")

    axes[2].imshow(img_test, cmap="viridis")
    axes[2].set_title("Test data")

    X = project_utils.generate_design_matrix(x, y, 2)

    print(X.shape)

    plt.show()

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


def part_b():

    x = np.arange(0, 1, STEP_SIZE)
    y = np.arange(0, 1, STEP_SIZE)

    x, y = np.meshgrid(x, y)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)
    franke_z_flat = franke_z.ravel()

    x = np.ravel(x)
    y = np.ravel(y)

    index_vals = np.array(range(np.prod(franke_z.shape)))

    idx_train, idx_test = train_test_split(
        index_vals, test_size=0.3, random_state=42)

    # print("Size of the test and training data sets:")

    # print(x_test.shape, x_train.shape, y_test.shape,
    #      y_train.shape, z_test.shape, z_train.shape)

    # Set up pandas frame for keeping scare of the results

    results = pd.DataFrame(
        columns=["Polynomial", "MSE_train", "MSE_test", "R2_train", "R2_test", "Lambda"])

    for p in range(1, 9):

        print("Polynomial degree: ", p)

        X_train = project_utils.generate_design_matrix(
            x[idx_train], y[idx_train], p)
        X_test = project_utils.generate_design_matrix(
            x[idx_test], y[idx_test], p)

        z_test = franke_z_flat[idx_test]
        z_train = franke_z_flat[idx_train]

        print("Size of X-matrix for training and testing:")
        print(X_train.shape, X_test.shape)

        for lmb in np.logspace(-3, 3, 20):

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

    x, y = np.meshgrid(x, y)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)
    franke_z_flat = franke_z.ravel()

    x = np.ravel(x)
    y = np.ravel(y)

    index_vals = np.array(range(np.prod(franke_z.shape)))

    idx_train, idx_test = train_test_split(
        index_vals, test_size=0.3, random_state=42)

    results = pd.DataFrame(
        columns=["Polynomial", "MSE_train", "MSE_test", "R2_train", "R2_test", "Lambda"])

    for p in range(1, 6):

        X_train = project_utils.generate_design_matrix(
            x[idx_train], y[idx_train], p)
        X_test = project_utils.generate_design_matrix(
            x[idx_test], y[idx_test], p)

        z_train = franke_z_flat[idx_train]
        z_test = franke_z_flat[idx_test]

        for lmb in np.logspace(-3, 3, 20):
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


def part_d():
    print("Pen and paper")


def part_e(step_size=STEP_SIZE):

    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    x, y = np.meshgrid(x, y)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)
    franke_z_flat = franke_z.ravel()

    x = np.ravel(x)
    y = np.ravel(y)

    index_vals = np.array(range(np.prod(franke_z.shape)))

    idx_train, idx_test = train_test_split(
        index_vals, test_size=0.3, random_state=42)

    # x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
    #    x, y, franke_z, test_size=0.3)

    results = pd.DataFrame(
        columns=["Polynomial",
                 "Error", "Bias", "Variance"])

    for p in range(1, 9):

        X_test = project_utils.generate_design_matrix(
            x[idx_test], y[idx_test], p)
        z_test = franke_z_flat[idx_test]

        x_train = x[idx_train]
        y_train = y[idx_train]
        z_train = franke_z_flat[idx_train]

        k_bootstraps = 100
        n_samples = 20000

        print("The relativel size of the bootstrap samples: ",
              n_samples/len(x_train))

        bootstrap_MSE = np.zeros(k_bootstraps)
        bootstrap_bias = np.zeros(k_bootstraps)
        bootstrap_variance = np.zeros(k_bootstraps)

        for i in range(k_bootstraps):

            _x, _y, _z = resample(x_train,
                                  y_train,
                                  z_train,
                                  n_samples=n_samples)

            _X = project_utils.generate_design_matrix(_x, _y, p)

            _MSE, _R2, _Z, _beta = project_utils.OLS(_X, _z)

            z_pred = X_test.dot(_beta)

            MSE = mean_squared_error(z_test, z_pred)
            bias = np.mean((z_test - np.mean(z_pred))**2)
            variance = np.var(z_pred)

            bootstrap_MSE[i] = MSE
            bootstrap_bias[i] = bias
            bootstrap_variance[i] = variance

        result = {"Polynomial": p,
                  "Error": np.mean(bootstrap_MSE),
                  "Bias": np.mean(bootstrap_bias),
                  "Variance": np.mean(bootstrap_variance)}

        results = results._append(result, ignore_index=True)

        # Plot the results

    project_utils.plot_bias_variance_tradeoff(results,
                                              OUTPUT_DIR,
                                              "franke_bias_variance.png")

    print(results)


def part_f(step_size=STEP_SIZE, k=5):
    """
    k-fold cross validation

    Implementation so unclever that I cant
    have lifted it from somewhere.

    """

    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    franke_z = project_utils.FrankeFunction(x, y, add_noise=True)

    idx = np.arange(len(x))

    np.random.shuffle(idx)  # Shuffle the indices

    # Perform k-fold cross validation
    # with multuple approaches for the same set
    # of folds

    type_vec = ["OLS", "ridge", "lasso"]
    p_vec = np.arange(1, 9)

    result_frame = pd.DataFrame(columns=["MSE_train", "MSE_test", "R2_train",
                                            "R2_test", "Polynomial", "Lambda", "Type", "Folds"])
    
    for type in type_vec:
            
            if type == "OLS":
                lambda_vals = [0]
            elif type == "ridge":
                lambda_vals = np.logspace(-3, 3, 20)
            elif type == "lasso":
                lambda_vals = np.logspace(-3, 3, 20)
    
            for p in p_vec:
    
                for lambda_ in lambda_vals:
    
                    foo = project_utils.k_fold_cross_validation(idx=idx,
                                                                x=x,
                                                                y=y,
                                                                z=franke_z,
                                                                k=k,
                                                                type=type,
                                                                lmb=lambda_,
                                                                p=p)
    
                    result_frame = result_frame._append(foo, ignore_index=True)

    print(result_frame)

    # Plot the results

    # First the OLS

    result_frame_ols = result_frame[result_frame["Type"] == "OLS"]

    project_utils.plot_mse_and_r2(result_frame_ols,
                                    OUTPUT_DIR,
                                    "franke_OLS_mse_r2_kfold.png",
                                    "OLS")
    
    # Then the ridge

    result_frame_ridge = result_frame[result_frame["Type"] == "ridge"]

    project_utils.plot_mse_and_r2(result_frame_ridge,
                                    OUTPUT_DIR,
                                    "franke_ridge_mse_r2_kfold.png",
                                    "Ridge")
    
    # Then the lasso

    result_frame_lasso = result_frame[result_frame["Type"] == "lasso"]

    project_utils.plot_mse_and_r2(result_frame_lasso,
                                    OUTPUT_DIR,
                                    "franke_lasso_mse_r2_kfold.png",
                                    "Lasso")


    sys.exit()

    type = "ridge"
    lambda_ = 0.1
    p = 3

    foo = project_utils.k_fold_cross_validation(idx=idx,
                                                x=x,
                                                y=y,
                                                z=franke_z,
                                                k=k,
                                                type=type,
                                                lmb=lambda_,
                                                p=p)

    print(foo)

    sys.exit()

    for p in range(1, 6):

        print("Now doing polynomial degree: ", p)

        # Set up k-fold cross validation

        MSE_test_kfold = np.zeros(k)
        R2_test_kfold = np.zeros(k)
        MSE_train_kfold = np.zeros(k)
        R2_train_kfold = np.zeros(k)

        index_groups = np.array_split(idx, k)

        for g, i in zip(index_groups, range(k)):

            x_fold = x[g]
            y_fold = y[g]
            z_fold = franke_z[g]

            x_train = x[~g]
            y_train = y[~g]
            z_train = franke_z[~g]

            X_train = project_utils.generate_design_matrix(x_train, y_train, p)
            X_test = project_utils.generate_design_matrix(x_fold, y_fold, p)

            MSE_train, R2_train, z_tilde_train, beta_ols = project_utils.OLS(
                X_train, z_train)

            z_tilde_test = X_test.dot(beta_ols)

            MSE_test = project_utils.MSE(z_fold, z_tilde_test)
            R2_test = project_utils.R2(z_fold, z_tilde_test)

            MSE_test_kfold[i] = MSE_test
            R2_test_kfold[i] = R2_test
            MSE_train_kfold[i] = MSE_train
            R2_train_kfold[i] = R2_train

        print("MSE test: ", np.mean(MSE_test_kfold))
        print("R2 test: ", np.mean(R2_test_kfold))
        print("MSE train: ", np.mean(MSE_train_kfold))
        print("R2 train: ", np.mean(R2_train_kfold))

# Run the parts


if DO_PART_A:
    part_a()
if DO_PART_B:
    part_b()
if DO_PART_C:
    part_c()
if DO_PART_D:
    part_d()
if DO_PART_E:
    part_e(step_size=0.001)
if DO_PART_F:
    part_f(step_size=0.001, k=5)
