"""

This file has drafts of functions that are a work in progress

It is all a mess now

"""

"""

Script to perform analysis of the data from the first project.

"""


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
import utils as project_utils


output_dir = 'output_tmp'

STEP_SIZE = 0.01

x = np.arange(0, 1, STEP_SIZE)
y = np.arange(0, 1, STEP_SIZE)

x_m,y_m = np.meshgrid(x,y)

z = project_utils.FrankeFunction(x_m, y_m)

p = 2

X = project_utils.generate_design_matrix(x_m, y_m, p)

print(z.shape)
print(X.shape)



sys.exit()


def FrankeFunction(x, y, add_noise=False, sigma=0.1):

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    if add_noise:
        noise = np.random.normal(0, sigma, size=term1.shape)
        return term1 + term2 + term3 + term4 + noise
    else:
        return term1 + term2 + term3 + term4


def design_matrix(x, y, n):
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


def plot_franke(x, y, z):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def ilus_franke():

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)

    # Add normally distributed noise

    z_noisy = z + np.random.normal(0, 0.1, size=z.shape)

    plot_franke(x, y, z_noisy)
    plot_franke(x, y, z)

    plt.show()


def MSE(y, y_tilde):

    n = len(y)
    return np.sum((y - y_tilde)**2)/n


def error(y, y_tilde):

    return np.mean((y - y_tilde)**2)


def R2(y, y_tilde):

    n = len(y)
    return 1 - np.sum((y - y_tilde)**2)/np.sum((y - np.mean(y))**2)


def OLS(X, z):

    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
    z_tilde = X.dot(beta)
    return MSE(z, z_tilde), R2(z, z_tilde), z_tilde, beta


def part_a():

    # Part A

    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    franke_noisy = FrankeFunction(x, y,
                                  add_noise=True,
                                  sigma=0.1)

    # Split ito training and test data

    x_test, x_train, y_test, y_train, z_test, z_train = train_test_split(
        x, y, franke_noisy, test_size=0.5)

    print(x_test.shape, x_train.shape, y_test.shape,
          y_train.shape, z_test.shape, z_train.shape)

    # Z here now is the noisy Franke function

    # Set up plotting

    fig_beta = plt.figure()
    ax_beta = fig_beta.add_subplot(1, 1, 1)

    fig_mse = plt.figure()
    ax_mse = fig_mse.add_subplot(1, 1, 1)

    fig_r2 = plt.figure()
    ax_r2 = fig_r2.add_subplot(1, 1, 1)

    mse_vec = np.zeros(5)
    r2_vec = np.zeros(5)

    mse_test = np.zeros(5)
    r2_test = np.zeros(5)

    for p in range(1, 6):

        X_p1_train = design_matrix(x_train, y_train, p)
        X_p1_test = design_matrix(x_test, y_test, p)

        MSE_train, R2_train, z_tilde_train, beta = OLS(X_p1_train, z_train)

        MSE_test = MSE(z_test, X_p1_test @ beta)
        R2_test = R2(z_test, X_p1_test @ beta)

        mse_vec[p-1] = MSE_train
        r2_vec[p-1] = R2_train

        mse_test[p-1] = MSE_test
        r2_test[p-1] = R2_test

        ax_beta.plot(np.arange(len(beta)), beta, label='p = %d' % p)

        ax_beta.set_xlabel(r'$\beta$')
        ax_beta.set_ylabel('Value')

        plt.legend()

    fig_beta.savefig(os.path.join(output_dir, 'beta.png'))

    ax_mse.plot(np.arange(1, 6), mse_vec, 'o', label='MSE - training')
    ax_mse.plot(np.arange(1, 6), mse_test, 'o', label='MSE - test')
    ax_mse.set_xlabel('Polynomial degree')
    ax_mse.set_ylabel('MSE')

    plt.legend()

    fig_mse.savefig(os.path.join(output_dir, 'MSE.png'))

    ax_r2.plot(np.arange(1, 6), r2_vec, 'o', label=r'$R^2$ - training')
    ax_r2.plot(np.arange(1, 6), r2_test, 'o', label=r'$R^2$ - test')

    ax_r2.set_xlabel('Polynomial degree')
    ax_r2.set_ylabel(r'$R^2$')

    plt.legend()

    fig_r2.savefig(os.path.join(output_dir, 'R2.png'))

    print("OLS - MSE: ", mse_vec, mse_test)
    print("OLS - R2: ", r2_vec, r2_test)

# Part B


def part_b():

    # Part B - Ridge regression

    x = np.arange(0, 1, 0.0001)
    y = np.arange(0, 1, 0.0001)

    franke_noisy = FrankeFunction(x, y,
                                  add_noise=True,
                                  sigma=0.1)

    # Split ito training and test data

    x_test, x_train, y_test, y_train, z_test, z_train = train_test_split(
        x, y, franke_noisy, test_size=0.5)

    print(x_test.shape, x_train.shape, y_test.shape,
          y_train.shape, z_test.shape, z_train.shape)

    figure_single_p_r2 = plt.figure()
    ax_r2 = figure_single_p_r2.add_subplot(1, 1, 1)

    figure_single_p_mse = plt.figure()
    ax_mse = figure_single_p_mse.add_subplot(1, 1, 1)

    for p in range(1, 5):

        X_p_train = design_matrix(x_train, y_train, p)
        X_p_test = design_matrix(x_test, y_test, p)

        I = np.eye(X_p_train.shape[1])

        print(I.shape)

        nlambda = 100
        # lambda_vec = np.logspace(-5, 5, nlambda) # Try first searaing over a wide range of lambda values
        lambda_vec = [1e-2]

        print(lambda_vec)

        MSE_train = np.zeros(len(lambda_vec))
        R2_train = np.zeros(len(lambda_vec))
        R2_test = np.zeros(len(lambda_vec))
        MSE_test = np.zeros(len(lambda_vec))

        for i in range(len(lambda_vec)):

            lmb = lambda_vec[i]

            beta_ridge = np.linalg.inv(X_p_train.T.dot(
                X_p_train) + lmb*I).dot(X_p_train.T).dot(z_train)

            ypred_train = X_p_train.dot(beta_ridge)

            MSE_train[i] = MSE(z_train, ypred_train)
            R2_train[i] = R2(z_train, ypred_train)

            ypred_test = X_p_test.dot(beta_ridge)

            MSE_test[i] = MSE(z_test, ypred_test)
            R2_test[i] = R2(z_test, ypred_test)

        ax_r2.plot(np.log10(lambda_vec), R2_train,
                   label=r'$R^2$ - training' + ' p = %d' % p)
        ax_r2.plot(np.log10(lambda_vec), R2_test,
                   label=r'$R^2$ - test' + ' p = %d' % p)

        ax_r2.set_xlabel(r'$\lambda$')
        ax_r2.set_ylabel(r'$R^2$')

        figure_single_p_r2.legend()

        figure_single_p_r2.savefig(os.path.join(output_dir, 'ridge_r2.png'))

        ax_mse.plot(np.log10(lambda_vec), MSE_train,
                    label='MSE - training' + ' p = %d' % p)
        ax_mse.plot(np.log10(lambda_vec), MSE_test,
                    label='MSE - test' + ' p = %d' % p)

        ax_mse.set_xlabel(r'$\lambda$')

        figure_single_p_mse.legend()

        figure_single_p_mse.savefig(os.path.join(output_dir, 'ridge_MSE.png'))

    print(MSE_test, R2_test)

# Now with lasso, lets simply use the one from sklearn


def part_c():

    x = np.arange(0, 1, 0.0001)
    y = np.arange(0, 1, 0.0001)

    z_noisy = FrankeFunction(x, y, add_noise=True, sigma=0.01)

    x_test, x_train, y_test, y_train, z_test, z_train = train_test_split(x, y, z_noisy,
                                                                         test_size=0.3,
                                                                         random_state=42)

    # For a single p

    p = 4

    lambda_vec = np.logspace(-10, -5, 100)
    # lambda_vec = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-2]

    MSE_train = np.zeros(len(lambda_vec))
    R2_train = np.zeros(len(lambda_vec))

    MSE_test = np.zeros(len(lambda_vec))
    R2_test = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):

        lmb = lambda_vec[i]

        X_p_train = design_matrix(x_train, y_train, p)
        X_p_test = design_matrix(x_test, y_test, p)

        # When using 100000 - 0.97-0.98 is achieved
        clf = linear_model.Lasso(
            alpha=lmb, fit_intercept=False, max_iter=10000)
        clf.fit(X_p_train, z_train)

        ypred_train = clf.predict(X_p_train)
        ypred_test = clf.predict(X_p_test)

        MSE_train[i] = MSE(z_train, ypred_train)
        R2_train[i] = R2(z_train, ypred_train)

        MSE_test[i] = MSE(z_test, ypred_test)
        R2_test[i] = R2(z_test, ypred_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    print(MSE_train, MSE_test)
    print(R2_train, R2_test)

    ax.plot(np.log10(lambda_vec), MSE_train, label='MSE - training')
    ax.plot(np.log10(lambda_vec), MSE_test, label='MSE - test')

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('MSE')

    plt.legend()

    fig.savefig(os.path.join(output_dir, "lasso_mse_" + str(p) + ".png"))

    fig_r2 = plt.figure()
    ax = fig_r2.add_subplot(1, 1, 1)

    ax.plot(np.log10(lambda_vec), R2_train, label=r'$R^2$ - training')
    ax.plot(np.log10(lambda_vec), R2_test, label=r'$R^2$ - test')

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$R^2$')

    fig_r2.savefig(os.path.join(output_dir, "lasso_r2_" + str(p) + ".png"))

# Part c, but now with scaling


def part_c_with_scaling():

    x = np.arange(0, 1, 0.0001)
    y = np.arange(0, 1, 0.0001)

    z_noisy = FrankeFunction(x, y, add_noise=True, sigma=0.01)

    x_test, x_train, y_test, y_train, z_test, z_train = train_test_split(
        x, y, z_noisy, test_size=0.5)

    p = 4

    X_p_train = design_matrix(x_train, y_train, p)

    scaler = preprocessing.StandardScaler().fit(X_p_train)

    X_scaled = scaler.transform(X_p_train)

    lambda_vec = np.logspace(-8, -5, 100)

    MSE_train = np.zeros(len(lambda_vec))
    R2_train = np.zeros(len(lambda_vec))

    MSE_test = np.zeros(len(lambda_vec))
    R2_test = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):

        lmb = lambda_vec[i]

        clf = linear_model.Lasso(alpha=lmb, fit_intercept=False)
        clf.fit(X_scaled, z_train)

        ypred_train = clf.predict(X_scaled)

        MSE_train[i] = MSE(z_train, ypred_train)
        R2_train[i] = R2(z_train, ypred_train)

        ypred_test = clf.predict(scaler.transform(
            design_matrix(x_test, y_test, p)))

        MSE_test[i] = MSE(z_test, ypred_test)
        R2_test[i] = R2(z_test, ypred_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(np.log10(lambda_vec), MSE_train, label='MSE - training')
    ax.plot(np.log10(lambda_vec), MSE_test, label='MSE - test')

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('MSE')

    plt.legend()

    fig.savefig(os.path.join(
        output_dir, "lasso_mse_scaled_" + str(p) + ".png"))

    fig_r2 = plt.figure()
    ax = fig_r2.add_subplot(1, 1, 1)

    ax.plot(np.log10(lambda_vec), R2_train, label=r'$R^2$ - training')
    ax.plot(np.log10(lambda_vec), R2_test, label=r'$R^2$ - test')

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$R^2$')

    fig_r2.legend()

    fig_r2.savefig(os.path.join(
        output_dir, "lasso_r2_scaled_" + str(p) + ".png"))


def part_a_with_scaling():

    x = np.arange(0, 1, 0.0001)
    y = np.arange(0, 1, 0.0001)

    z_noisy = FrankeFunction(x, y, add_noise=True, sigma=0.01)

    x_test, x_train, y_test, y_train, z_test, z_train = train_test_split(
        x, y, z_noisy, test_size=0.5)

    p = 4

    X_p_train = design_matrix(x_train, y_train, p)
    X_p_test = design_matrix(x_test, y_test, p)

    scaler = preprocessing.StandardScaler().fit(X_p_train)

    X_scaled = scaler.transform(X_p_train)
    X_test_scaled = X_p_test

    beta_ols = np.linalg.pinv(X_scaled.T.dot(
        X_scaled)).dot(X_scaled.T).dot(z_train)

    ypred_train = X_scaled.dot(beta_ols)

    print(MSE(z_train, ypred_train), R2(z_train, ypred_train))


def part_e():

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    z_noisy = FrankeFunction(x, y, add_noise=True, sigma=0.1)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z_noisy,
                                                                         test_size=0.3)

    max_degree = 21

    MSE_train_vec = np.zeros(max_degree-1)
    MSE_test_vec = np.zeros(max_degree-1)

    for p in range(1, max_degree):

        X_p1_train = design_matrix(x_train, y_train, p)
        X_p1_test = design_matrix(x_test, y_test, p)

        MSE_train, R2_train, z_tilde_train, beta = OLS(X_p1_train, z_train)

        MSE_test_vec[p-1] = MSE(z_test, X_p1_test @ beta)
        MSE_train_vec[p-1] = MSE_train

    ax.plot(np.arange(1, max_degree), MSE_train_vec,
            'o', label='MSE - training')
    ax.plot(np.arange(1, max_degree), MSE_test_vec, 'o', label='MSE - test')

    ax.set_xlabel('Complexicity')
    ax.set_ylabel('MSE')

    plt.legend()

    fig.savefig(os.path.join(output_dir, "figure_2_11.png"))

    # Bootstrapig

    print(len(x_train), len(y_train), len(z_train))
    print(len(x_test), len(y_test), len(z_test))

    n_samples = 20
    k_bootstraps = 5

    MSE_test_vec = np.zeros(max_degree-1)

    error_vec = np.zeros(max_degree-1)
    bias_vec = np.zeros(max_degree-1)
    variance_vec = np.zeros(max_degree-1)

    for p in range(1, max_degree-1):

        X_p1_train = design_matrix(x_train, y_train, p)
        X_p1_test = design_matrix(x_test, y_test, p)

        error_vec_bootstrap = np.zeros(k_bootstraps)
        bias_vec_bootstrap = np.zeros(k_bootstraps)
        variance_vec_bootstrap = np.zeros(k_bootstraps)

        for k in range(k_bootstraps):

            x_sample, y_sample, z_sample = resample(
                x_train, y_train, z_train, n_samples=n_samples)

            X_p1_sample = design_matrix(x_sample, y_sample, p)

            MSE_train, R2_train, z_tilde_train, beta = OLS(
                X_p1_sample, z_sample)

            z_pred = X_p1_test @ beta

            error_vec_bootstrap[k] = error(z_test, X_p1_test @ beta)
            bias_vec_bootstrap[k] = np.mean((z_test - np.mean(z_pred))**2)
            variance_vec_bootstrap[k] = np.mean(np.var(z_pred))

        error_vec[p-1] = np.mean(error_vec_bootstrap)
        bias_vec[p-1] = np.mean(bias_vec_bootstrap)
        variance_vec[p-1] = np.mean(variance_vec_bootstrap)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(np.arange(1, max_degree), error_vec, 'o', label='Error')
    ax.plot(np.arange(1, max_degree), bias_vec, 'o', label='Bias')
    ax.plot(np.arange(1, max_degree), variance_vec, 'o', label='Variance')

    ax.set_xlabel('Complexicity')
    ax.set_ylabel('Error')

    plt.legend()

    fig.savefig(os.path.join(output_dir, "error_bias_variance.png"))
