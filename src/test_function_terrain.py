"""

This script is used to test the OLS method on the terrain data.
Various stress tests are performed to test whether downsampling
of the data or smoothing with a Gaussian filter 

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
from skimage.transform import resize
import time
from scipy.ndimage import gaussian_filter

# Global plotting parameters

sns.set_style("whitegrid")
sns.set_context("paper")

# Global variables

SIGMA = 0.1  # Noise level
OUTPUT_DIR = "output_tmp"  # Output directory for figures
STEP_SIZE = 0.01  # Step size for sampling the Franke function
DPI_FIG = project_utils.DPI_FIG  # DPI for saving figures
PERFORM_SCALING = True  # Perform scaling of the data
PERFORM_SMOOTHING = True
PLOT_TIME = False

# Running flags

DO_PART_A = True # Only implemented for OLS

def part_a(plot_prediction=False):

    # Load the terrain data

    terrain = list(project_utils.get_terrain_data())

    # Use only the first terrain data set

    data, name = terrain[0]

    data_orig = data.copy()

    # Downsample the data

    data = resize(data, (int(data.shape[0]/12), int(data.shape[1]/12)), anti_aliasing=True)

    # Perform a gaussian smoothing of the data
    
    if PERFORM_SMOOTHING:
        data = gaussian_filter(data, sigma=10)

    print(data.shape, name)

    x = np.arange(0, data.shape[0])
    y = np.arange(0, data.shape[1])

    x, y = np.meshgrid(x, y)

    data_flat = data.ravel()
    x = x.ravel()
    y = y.ravel()
    
    index_vals = np.array(range(np.prod(data.shape)))

    print(index_vals.shape)

    idx_train, idx_test = train_test_split(
        index_vals, test_size=0.3, random_state=42)
    
    if PERFORM_SCALING:

        scaler_vals = preprocessing.StandardScaler()
        coord_scaler_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        coord_scaler_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        x_orig = x.copy()
        y_orig = y.copy()
        franke_z_flat_orig = data_flat.copy()

        #x[idx_train] = coord_scaler_x.fit_transform(x[idx_train].reshape(-1, 1)).ravel()
        #y[idx_train] = coord_scaler_y.fit_transform(y[idx_train].reshape(-1, 1)).ravel()
        data_flat[idx_train] = scaler_vals.fit_transform(data_flat[idx_train].reshape(-1, 1)).ravel()
        data_flat[idx_test] = scaler_vals.transform(data_flat[idx_test].reshape(-1, 1)).ravel()

    # Save the train and test data sets as images

    project_utils.plot_train_test_image(
        data, idx_test, idx_train, OUTPUT_DIR, "test_terrain_training_test.png")
    
    plt.close()

    results = pd.DataFrame(
        columns=["Polynomial", "MSE_test", "MSE_train", "R2_test", "R2_train"])
        
    if plot_prediction:

        # Only plot the prediction for these polynomial degrees

        p_to_plot = [5, 20, 50]

        fig, axes = plt.subplots(nrows=len(p_to_plot)+1, ncols=2, figsize=(5, 10))

        print(axes)

        axes[0,0].imshow(data, cmap="viridis")
        axes[0,0].set_title("Data")

        plot_row = 1

    elapsed_time = []

    for p in range(1, 71, 4):
    #for p in p_to_plot:

        start_time = time.time()
        
        print(p)

        X_train = project_utils.generate_design_matrix(
            x[idx_train], y[idx_train], p)
        X_test = project_utils.generate_design_matrix(
            x[idx_test], y[idx_test], p)

        MSE_train, R2_train, z_pred_train, beta = project_utils.OLS(
            X_train, data_flat[idx_train])

        z_pred = X_test.dot(beta)
        z_test = data_flat[idx_test]

        z_train = data_flat[idx_train]

        MSE_test = project_utils.MSE(z_test, z_pred)

        R2_test = project_utils.R2(z_test, z_pred)

        result = {"Polynomial": p, "MSE_test": MSE_test,
                  "MSE_train": MSE_train, "R2_test": R2_test, "R2_train": R2_train}

        results = results._append(result, ignore_index=True)

        end_time = time.time()

        elapsed_time.append(end_time - start_time)

        if plot_prediction and p in p_to_plot:

            X_whole_image = project_utils.generate_design_matrix(x, y, p)

            Z_pred_whole_image = X_whole_image.dot(beta)

            Z_pred_whole_image = Z_pred_whole_image.reshape(data.shape)

            img_predicton = np.zeros(index_vals.shape)

            img_predicton[idx_test] = z_pred

            img_predicton = img_predicton.reshape(data.shape)
            
            axes[plot_row,0].imshow(Z_pred_whole_image, cmap="viridis")
            axes[plot_row,0].set_title("Prediction for p = " + str(p))

            axes[plot_row,1].imshow(
                (data - Z_pred_whole_image.reshape(data.shape)), cmap="viridis")
            axes[plot_row,1].set_title("Difference for p = " + str(p))

            plot_row += 1

            print(plot_row)


    if plot_prediction:
        for ax in axes.ravel():
            ax.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTPUT_DIR, "test_terrain_prediction_all" + ".png"), dpi = DPI_FIG)

    print(results)

    print("Time elapsed: ", elapsed_time)

    if PLOT_TIME:

        fig_time = plt.figure()
        ax = fig_time.add_subplot(111)

        ax.plot(range(1,71,4), elapsed_time, label="Time elapsed")

        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("Time elapsed [s]")

        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTPUT_DIR, "test_terrain_time" + ".png"), dpi = DPI_FIG)

    project_utils.plot_mse_and_r2(results,
                                  OUTPUT_DIR,
                                  "test_terrain_OLS_mse_r2.png",
                                  "OLS")
    
if DO_PART_A:
    part_a(plot_prediction=True)