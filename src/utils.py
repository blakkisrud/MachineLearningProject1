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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from random import random, seed
import sys

from sklearn.model_selection import train_test_split, KFold
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.utils import resample

import os
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

# Functions

def ilus_franke(step_size = 0.01, noise = 0.1, plot = True, seed = 42):

    # Make data.
    x = np.arange(0, 1, step_size)
    y = np.arange(0, 1, step_size)

    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)

    # Add normally distributed noise

    z_noisy = z + np.random.normal(0, noise, size=z.shape)

    plot_franke(x, y, z_noisy)
    plot_franke(x, y, z)

    plt.show()

def plot_franke(x, y, z):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig


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

def plot_train_test_image(z, idx_test, idx_train, output_dir, filename):

    """

    z here is a 2d array

    """

    z_flat = np.ravel(z)

    img_train = np.zeros((len(z_flat)))
    img_test = np.zeros((len(z_flat)))

    img_train[idx_train] = z_flat[idx_train]
    img_test[idx_test] = z_flat[idx_test]

    img_train = np.reshape(img_train, z.shape)
    img_test = np.reshape(img_test, z.shape)

    fig = plt.figure()

    ax = fig.add_subplot(1,3,1)
    ax.imshow(z, cmap="viridis")

    ax = fig.add_subplot(1,3,2)
    ax.imshow(img_train, cmap="viridis")

    ax = fig.add_subplot(1,3,3)
    ax.imshow(img_test, cmap="viridis")

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, filename))

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
            result_frame["Bias"], '-o',
            label="Bias")
    
    ax.plot(result_frame["Polynomial"],
            result_frame["Variance"], '-o',
            label="Variance")
    
    ax.plot(result_frame["Polynomial"],
            result_frame["Error"], '-o',
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
    
def k_fold_cross_validation(idx, k, type, p, x, y, z, lmb = None):

    """
    K-fold cross validation, idx is a list of indices.
    in random order, k is the number of folds.
    Including also the polynomial degree p for bookeeping purposes.

    Type is a string that decides if either OLS, Ridge or LASSO
    is used

    Should return a pandas frame with the MSE, p2 of train
    and test of each fold.

    """

    if type == "ridge" and lmb == None:
        raise ValueError("Lambda must be specified for ridge regression.")
    
    if type == "lasso" and lmb == None:
        raise ValueError("Lambda must be specified for LASSO regression.")

    index_groups = np.array_split(idx, k)

    result_frame = pd.DataFrame(columns=["MSE_train", "MSE_test", "R2_train", 
                                         "R2_test", "Polynomial", "Lambda", "Type"])
    
    MSE_test_kfold = np.zeros(k)
    MSE_train_kfold = np.zeros(k)
    R2_test_kfold = np.zeros(k)
    R2_train_kfold = np.zeros(k)

    for g, i in zip(index_groups, range(k)):

        x_fold = x[g]
        y_fold = y[g]
        z_fold = z[g]

        x_train = x_train[~g]
        y_train = y_train[~g]
        z_train = z_train[~g]

        X_train = generate_design_matrix(x_train, y_train, p)
        X_test = generate_design_matrix(x_fold, y_fold, p)

        if type == "OLS":
            MSE_train, R2_train, z_tilde_train, beta_train = OLS(X_train, z_train)
            z_tilde_test = X_test.dot(beta_train)

        elif type == "ridge":
            MSE_train, R2_train, z_tilde_train, beta_train = ridge(X_train, z_train, lmb)
            z_tilde_test = X_test.dot(beta_train)

        elif type == "lasso":
            clf = linear_model.Lasso(alpha=lmb, fit_intercept=False)
            clf.fit(X_train, z_train)
            z_tilde_train = clf.predict(X_train)
            z_tilde_test = clf.predict(X_test)

            MSE_train = MSE(z_train, z_tilde_train)
            R2_train = R2(z_train, z_tilde_train)

        MSE_test = MSE(z_fold, z_tilde_test)
        R2_test = R2(z_fold, z_tilde_test)

        MSE_test_kfold[i] = MSE_test
        MSE_train_kfold[i] = MSE_train
        R2_test_kfold[i] = R2_test
        R2_train_kfold[i] = R2_train

    result_frame["MSE_train"] = np.mean(MSE_train_kfold)
    result_frame["MSE_test"] = np.mean(MSE_test_kfold)
    result_frame["R2_train"] = np.mean(R2_train_kfold)
    result_frame["R2_test"] = np.mean(R2_test_kfold)
    result_frame["Polynomial"] = p
    result_frame["Lambda"] = lmb
    result_frame["Type"] = type

    return result_frame




def get_terrain_data():
    # add all terrain data, with names, here
    from imageio.v3 import imread

    for terrain, TERRAIN_NAME in zip([imread("src/SRTM_data_Norway_1.tif"), imread("src/SRTM_data_Norway_2.tif"), imread("src/oslo.tif"), imread("src/eide.tif")],
                                    ["terrain1", "terrain2", "oslo", "eide"]):
        yield terrain, TERRAIN_NAME


class TerrainAnalyser():
    def __init__(self, terrain, terrain_name="noname",
                 output_dir="", calculate_scores=True, split_on_chunk=False,
                 k_val_split=3, avals=np.logspace(0, 3, 4), pvals=list(range(2)),
                 verbose=True):

        self.terrain = terrain
        self.terrain_name = terrain_name

        self.index_vals = np.array(range(np.prod(terrain.shape)))
        self.terrain_values = self.terrain.flatten()
        self.split_on_chunk = split_on_chunk        # Holds-out partition of image for test if True, else makes random pixels test
        self.verbose = verbose

        self.test_set_ratio = 0.20
        self.random_state = 42

        self.make_train_test()

        # Save original train / test values for plotting histogram BEFORE and AFTER normalization
        self.values_train_orig = self.values_train
        self.values_test_orig = self.values_test
        self.normalize()

        self.output_dir = output_dir
        self.path_scores_train = os.path.join(self.output_dir, f"mse_{'''chunk''' if self.split_on_chunk else '''pixel'''}_split_{self.terrain_name}_train.npy")
        self.path_scores_val = os.path.join(self.output_dir, f"mse_{'''chunk''' if self.split_on_chunk else '''pixel'''}_split_{self.terrain_name}_val.npy")


        self.calculate_scores = calculate_scores    # Loads validation-scores, for HP tuning of alpha and p, from .npy file if false

        self.k_val_split = k_val_split  # number of validation splits, of training set, using k-fold
        self.avals = avals
        self.pvals = pvals

        self.model_func = linear_model.Ridge


    def make_train_test(self):
        if self.split_on_chunk:
            self.idx_train = self.index_vals[:int(len(self.index_vals) * (1 - self.test_set_ratio))]
            self.idx_test = self.index_vals[max(self.idx_train) + 1:]

        else:
            self.idx_train, self.idx_test = train_test_split(self.index_vals, test_size=self.test_set_ratio, random_state=self.random_state)


        self.values_train = self.terrain_values[self.idx_train]
        self.values_test = self.terrain_values[self.idx_test]



        self.x_train = self.idx_train.reshape(-1, 1) % self.terrain.shape[0]
        self.y_train = self.idx_train.reshape(-1, 1) // self.terrain.shape[0]
        self.x_test = self.idx_test.reshape(-1, 1) % self.terrain.shape[0]
        self.y_test = self.idx_test.reshape(-1, 1) // self.terrain.shape[0]


        if self.verbose:
            print(f"\tcreated train / test sets from {'''image partitions''' if self.split_on_chunk else '''random pixels'''}", end=" ")
            print("of sizes", self.idx_train.shape, self.idx_test.shape)
        return 1


    def normalize(self, plot_histogram=False):
        # Standard-score scaling image gray-levels
        # Min-max scaling coordinates to -1, 1
        # Fit to train data, only transform test

        self.scaler_img = preprocessing.StandardScaler()
        self.scaler_x = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))


        self.values_train = self.scaler_img.fit_transform(self.values_train.reshape(-1, 1))
        self.values_test = self.scaler_img.transform(self.values_test.reshape(-1, 1))

        self.x_train = self.scaler_x.fit_transform(self.x_train).reshape(-1)
        self.y_train = self.scaler_y.fit_transform(self.y_train).reshape(-1)

        self.x_test = self.scaler_x.transform(self.x_test).reshape(-1)
        self.y_test = self.scaler_y.transform(self.y_test).reshape(-1)

        if self.verbose:
            print("\timage data normalized by standard-scaling, xy-coordinates by minmax-scaling, using values from training data")

        return 1


    def show_terrain_partitions(self, showplt=True):
        # TODO: include showing validation partitions here
        splitmode = "chunk" if self.split_on_chunk else "pixel" # for naming

        # plot_train_test_image(self.terrain, self.idx_test, self.idx_train, output_dir=self.output_dir, filename=f"train_test_split_{self.terrain_name}_{splitmode}")


        terrain_shape = self.terrain.shape
        fig, ax = plt.subplots(ncols=self.k_val_split)

        for ki, idx_train_k, idx_val_k, values_train_k, values_val_k, x_train_k, x_val_k, y_train_k, y_val_k in self.get_validation_set(return_index=True):
            img_val = np.empty(shape=np.prod(terrain_shape))

            img_val[idx_val_k] = values_val_k.reshape(-1)
            img_val = img_val.reshape(terrain_shape)

            ax[ki].imshow(img_val)
            ax[ki].set_title(f"Fold {ki + 1}")
            ax[ki].axis("off")


        img_train = np.empty(shape=np.prod(terrain_shape))
        img_test = np.empty(shape=np.prod(terrain_shape))

        img_train[self.idx_train] = self.values_train.ravel()
        img_test[self.idx_test] = self.values_test.ravel()

        img_train = img_train.reshape(terrain_shape)
        img_test = img_test.reshape(terrain_shape)

        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(img_train)
        ax[0].set_title("Train")
        ax[1].imshow(img_test)
        ax[1].set_title("Test")
        ax[2].imshow(img_train + img_test)
        ax[2].set_title("Train + test")
        [axi.axis("off") for axi in ax]


        plt.show() if showplt else 0


        return 1


    def get_validation_set(self, return_index=False):
        num_px_per_fold = len(self.x_train) // self.k_val_split

        if not self.split_on_chunk:

            # Splits by chunk if shuffle = False :))
            kfold_splitter = KFold(n_splits=self.k_val_split, random_state=self.random_state, shuffle=True)
            kfold_splits = kfold_splitter.split(X=self.idx_train)


        for ki in range(self.k_val_split):
            if self.split_on_chunk:
                idx_val_k = self.idx_train[num_px_per_fold * ki:len(self.idx_train) - num_px_per_fold * (self.k_val_split - ki - 1)]
                idx_train_k = np.setdiff1d(self.idx_train, idx_val_k)
            else:
                # print(num_px_per_fold, len(self.x_train) - num_px_per_fold)
                idx_train_k, idx_val_k = next(kfold_splits)
                # print(idx_train_k.shape, idx_val_k.shape)

            values_train_k = self.scaler_img.transform(self.terrain_values[idx_train_k].reshape(-1, 1))
            values_val_k = self.scaler_img.transform(self.terrain_values[idx_val_k].reshape(-1, 1))

            print(f"\n{self.k_val_split}-fold cross-validation: ki={ki}")
            print(idx_train_k.shape, idx_val_k.shape)

            x_train_k = idx_train_k.reshape(-1, 1) % self.terrain.shape[0]
            y_train_k = idx_train_k.reshape(-1, 1) // self.terrain.shape[0]
            x_val_k = idx_val_k.reshape(-1, 1) % self.terrain.shape[0]
            y_val_k = idx_val_k.reshape(-1, 1) // self.terrain.shape[0]

            x_train_k = self.scaler_x.transform(x_train_k).reshape(-1)
            y_train_k = self.scaler_y.transform(y_train_k).reshape(-1)
            x_val_k = self.scaler_x.transform(x_val_k).reshape(-1)
            y_val_k = self.scaler_y.transform(y_val_k).reshape(-1)

            if return_index:
                yield (ki, idx_train_k, idx_val_k,
                       values_train_k, values_val_k, x_train_k, x_val_k, y_train_k, y_val_k)
            else:
                yield ki, values_train_k, values_val_k, x_train_k, x_val_k, y_train_k, y_val_k


    def calculate_k_fold_cross_validated(self):

        mse_train_scores = np.ones(shape=(self.k_val_split, len(self.pvals), len(self.avals)))
        mse_val_scores = np.ones(shape=(self.k_val_split, len(self.pvals), len(self.avals)))

        for ki, values_train_k, values_val_k, x_train_k, x_val_k, y_train_k, y_val_k in self.get_validation_set():

            for pi, p in enumerate(self.pvals):

                X_train_k = generate_design_matrix(x_train_k, y_train_k, n=p)
                X_val_k = generate_design_matrix(x_val_k, y_val_k, n=p)
                print(X_train_k.shape, X_val_k.shape)

                for ai, a in enumerate(self.avals):
                    print(f"p={p}, a={a:.2e}", end="\t")
                    m = self.model_func(fit_intercept=False, alpha=a)
                    m.fit(X_train_k, values_train_k)

                    zhat_train = m.predict(X_train_k)
                    zhat_val = m.predict(X_val_k)

                    r2_train, r2_val = r2_score(values_train_k, zhat_train), r2_score(values_val_k, zhat_val)
                    mse_train, mse_val = mean_squared_error(values_train_k, zhat_train), mean_squared_error(
                        values_val_k, zhat_val)
                    print(f"train / val r2 = {r2_train:.3g} / {r2_val:.3g} \t mse = {mse_train:.3g} / {mse_val:.3g}")

                    mse_train_scores[ki, pi, ai] = mse_train
                    mse_val_scores[ki, pi, ai] = mse_val
                    # scores.loc[]

        np.save(self.path_scores_train, mse_train_scores)
        np.save(self.path_scores_val, mse_val_scores)

        print("SAVED TRAIN / VALIDATION SCORES IN ", self.output_dir)

        return 1


    def load_k_fold_cross_validated(self):
        # mse_train_scores = np.load(os.path.join(OUTPUT_DIR, f"mse_{'''chunk''' if SPLIT_ON_CHUNK else '''pixel'''}_split_{TERRAIN_NAME}_train.npy"), allow_pickle=True)
        # mse_val_scores = np.load(os.path.join(OUTPUT_DIR, f"mse_{'''chunk''' if SPLIT_ON_CHUNK else '''pixel'''}_split_{TERRAIN_NAME}_val.npy"), allow_pickle=True)
        self.mse_train_scores = np.load(self.path_scores_train, allow_pickle=True)
        self.mse_val_scores = np.load(self.path_scores_val, allow_pickle=True)

        self.mse_train_mean = np.mean(self.mse_train_scores, axis=0)
        self.mse_val_mean = np.mean(self.mse_val_scores, axis=0)

        print("LOADED TRAIN / VALIDATION SCORES FROM ", self.output_dir, self.mse_train_scores.shape, self.mse_val_scores.shape)
        return self.mse_train_scores, self.mse_val_scores


    def show_k_fold_cross_validated_scores(self, showplt=True):
        # must be run after loader / calculator for obvious reasons

        vmin = np.min([self.mse_train_mean, self.mse_val_mean])

        fig, ax = plt.subplots(ncols=2, sharey=True)
        i = 0

        for vals, name in zip([self.mse_train_mean, self.mse_val_mean], ["Train", "Val"]):
            # vmin = np.min(vals)
            # vmax = np.max(vals)
            im = ax[i].imshow(vals.T, vmin=vmin, vmax=1, cmap="RdYlGn_r")

            # TODO: fix colobars
            divider = make_axes_locatable(ax[i])
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # fig.colorbar(im, ax[i])

            ax[i].set_title(name)
            ax[i].set_xlabel("p")
            ax[i].set_ylabel("alpha")
            ax[i].set_yticks(list(range(len(self.avals))), [f"{a:.2e}" for a in self.avals])
            ax[i].set_xticks(list(range(len(self.pvals))), self.pvals)
            ax[i].grid(0)
            i += 1

        fig.tight_layout()
        fig.suptitle(self.terrain_name)
        plt.savefig(os.path.join(self.output_dir,
                                 f"cv_mse_{'''chunk''' if self.split_on_chunk else '''pixel'''}_split_{self.terrain_name}"))
        plt.show() if showplt else 0

        return 1


    def find_optimal_hps(self):
        # Find MSE minima - optimal alpha, p

        idx_opt = np.unravel_index(np.argmin(self.mse_val_mean, axis=None), self.mse_val_mean.shape)
        self.p_opt, self.alpha_opt = self.pvals[idx_opt[0]], self.avals[idx_opt[1]]
        print(f"Optimal p={self.p_opt}, alpha={self.alpha_opt:.3e} (minimal validation MSE = {np.min(self.mse_val_mean):.3e})")

        return self.p_opt, self.alpha_opt


    def evaluate_on_test(self):
        # Test on hold-out data
        m = self.model_func(fit_intercept=False, alpha=self.alpha_opt)
        X_train = generate_design_matrix(self.x_train, self.y_train, n=self.p_opt)
        X_test = generate_design_matrix(self.x_test, self.y_test, n=self.p_opt)

        m.fit(X_train, self.values_train)
        zhat_train = m.predict(X_train)
        zhat_test = m.predict(X_test)

        r2_train, r2_test = r2_score(self.values_train, zhat_train), r2_score(self.values_test, zhat_test)
        mse_train, mse_test = mean_squared_error(self.values_train, zhat_train), mean_squared_error(self.values_test, zhat_test)
        print(f"R2 train={r2_train:.3e}, test={r2_test:.3e}\tMSE train={mse_train:.3e}, test={mse_test:.3e}")


        return 1
