from utils import *
import pandas as pd
import numpy as np
from imageio.v3 import imread


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import resample

# from utils import generate_design_matrix
OUTPUT_DIR = "output_tmp"

# terrain = imread("SRTM_data_Norway_1.tif"); TERRAIN_NAME = "terrain1"
# terrain = imread("SRTM_data_Norway_2.tif"); TERRAIN_NAME = "terrain2"
terrain = imread("oslo.tif"); TERRAIN_NAME = "oslo"
# terrain = imread("eide.tif"); TERRAIN_NAME = "eide"


# TODO: average across geographic data - model generalization?


CALCULATE_SCORES = True   # Set to false to load pre-calculated scores


k = 3
avals = np.logspace(0, 5, 32)
pvals = [0, 1, 2, 3, 4, 5]


# plt.figure()
# plt.imshow(terrain)

# from draft_source import plot_vispy_terrain
# plot_vispy_terrain(terrain)

# plt.show()


index_vals = np.array(range(np.prod(terrain.shape)))
terrain_values = terrain.flatten()


# Create hold-out test set by partitioning bottom 20% of image
idx_train = index_vals[:int(len(index_vals) * 0.80)]
idx_test = index_vals[max(idx_train)+1:]


x_train = idx_train.reshape(-1, 1) % terrain.shape[0]
y_train = idx_train.reshape(-1, 1) // terrain.shape[0]
x_test = idx_test.reshape(-1, 1) % terrain.shape[0]
y_test = idx_test.reshape(-1, 1) // terrain.shape[0]

# Standard-scale pixel gray-values, minmax scale coordinates
scaler_img = StandardScaler()
values_train = scaler_img.fit_transform(terrain_values[idx_train].reshape(-1, 1))
values_test = scaler_img.transform(terrain_values[idx_test].reshape(-1, 1))


scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler_x.fit_transform(x_train).reshape(-1)
x_test = scaler_x.transform(x_test).reshape(-1)
y_train = scaler_y.fit_transform(y_train).reshape(-1)
y_test = scaler_y.transform(y_test).reshape(-1)


# show_terrain_partitions(terrain.shape, idx_train, idx_test, values_train, values_test)


# Divide train set into K folds for hyperparameter-tuning using cross-validation
print(x_train.shape, y_train.shape)


def calculate_k_fold_cross_validated(k, avals, pvals):
    num_px_per_fold = len(x_train) // k
    print(num_px_per_fold, len(x_train) - num_px_per_fold)

    mse_train_scores = np.ones(shape=(k, len(pvals), len(avals)))
    mse_val_scores = np.ones(shape=(k, len(pvals), len(avals)))

    for ki in range(k):
        idx_val_k = idx_train[num_px_per_fold * ki:len(idx_train) - num_px_per_fold*(k-ki-1)]
        idx_train_k = np.setdiff1d(idx_train, idx_val_k)


        values_train_k = scaler_img.transform(terrain_values[idx_train_k].reshape(-1, 1))
        values_val_k = scaler_img.transform(terrain_values[idx_val_k].reshape(-1, 1))


        show_terrain_partitions(terrain.shape, idx_train_k, idx_val_k, values_train_k, values_val_k)

        print(f"\n{k}-fold cross-validation: ki={ki}")
        print(idx_train_k.shape, idx_val_k.shape)


        x_train_k = idx_train_k.reshape(-1, 1) % terrain.shape[0]
        y_train_k = idx_train_k.reshape(-1, 1) // terrain.shape[0]
        x_val_k = idx_val_k.reshape(-1, 1) % terrain.shape[0]
        y_val_k = idx_val_k.reshape(-1, 1) // terrain.shape[0]

        x_train_k = scaler_x.transform(x_train_k).reshape(-1)
        x_val_k = scaler_x.transform(x_val_k).reshape(-1)
        y_train_k = scaler_y.transform(y_train_k).reshape(-1)
        y_val_k = scaler_y.transform(y_val_k).reshape(-1)

        p = 3

        for pi, p in enumerate(pvals):

            X_train_k = generate_design_matrix(x_train_k, y_train_k, n=p)
            X_val_k = generate_design_matrix(x_val_k, y_val_k, n=p)
            print(X_train_k.shape, X_val_k.shape)


            for ai, a in enumerate(avals):
                print(f"p={p}, a={a:.2e}", end="\t")
                m = Ridge(fit_intercept=False, alpha=a)
                m.fit(X_train_k, values_train_k)

                zhat_train = m.predict(X_train_k)
                zhat_val = m.predict(X_val_k)

                r2_train, r2_val = r2_score(values_train_k, zhat_train), r2_score(values_val_k, zhat_val)
                mse_train, mse_val = mean_squared_error(values_train_k, zhat_train), mean_squared_error(values_val_k, zhat_val)
                print(f"train / val r2 = {r2_train:.3g} / {r2_val:.3g} \t mse = {mse_train:.3g} / {mse_val:.3g}")


                mse_train_scores[ki, pi, ai] = mse_train
                mse_val_scores[ki, pi, ai] = mse_val
                # scores.loc[]

    # mse_train_scores.tofile(os.path.join(OUTPUT_DIR, "mse_train_scores.npy"))
    # mse_val_scores.tofile(os.path.join(OUTPUT_DIR, "mse_val_scores.npy"))
    np.save(os.path.join(OUTPUT_DIR, f"{TERRAIN_NAME}_mse_train_scores.npy"), mse_train_scores)
    np.save(os.path.join(OUTPUT_DIR, f"{TERRAIN_NAME}_mse_val_scores.npy"), mse_val_scores)
    print("SAVED TRAIN / VALIDATION SCORES IN ", OUTPUT_DIR)
    pass


def load_k_fold_cross_validated():
    mse_train_scores = np.load(os.path.join(OUTPUT_DIR, f"{TERRAIN_NAME}_mse_train_scores.npy"), allow_pickle=True)
    mse_val_scores = np.load(os.path.join(OUTPUT_DIR, f"{TERRAIN_NAME}_mse_val_scores.npy"), allow_pickle=True)
    return mse_train_scores, mse_val_scores


plt.close()


if CALCULATE_SCORES:
    calculate_k_fold_cross_validated(k, avals, pvals)


mse_train, mse_val = load_k_fold_cross_validated()
print(mse_train.shape, mse_val.shape)


mse_train_mean, mse_train_std = np.mean(mse_train, axis=0), np.std(mse_train, axis=0)
mse_val_mean, mse_val_std = np.mean(mse_val, axis=0), np.std(mse_val, axis=0)
print(mse_train_mean.shape)


print(mse_val_mean.shape)

vmin = np.min([mse_train_mean, mse_val_mean])

fig, ax = plt.subplots(ncols=2, sharey=True)
i = 0
for vals, name in zip([mse_train_mean, mse_val_mean], ["Train", "Val"]):
    # vmin = np.min(vals)
    # vmax = np.max(vals)
    ax[i].imshow(vals.T, vmin=vmin, vmax=1, cmap="RdYlGn_r")

    # plt.imshow(np.log(vals.T))
    ax[i].set_title(name)
    ax[i].set_xlabel("p")
    ax[i].set_ylabel("alpha")
    ax[i].set_yticks(list(range(len(avals))), [f"{a:.2e}" for a in avals])
    ax[i].set_xticks(list(range(len(pvals))), pvals)
    ax[i].grid(0)
    i += 1

plt.savefig(os.path.join(OUTPUT_DIR, f"{TERRAIN_NAME}_mse_cv.png"))


# Find minima - optimal alpha, p
print(np.argmin(mse_val_mean))
idx_opt = np.unravel_index(np.argmin(mse_val_mean, axis=None), mse_val_mean.shape)
print(idx_opt)
p_opt, alpha_opt = pvals[idx_opt[0]], avals[idx_opt[1]]
print(p_opt, alpha_opt)


# Test on hold-out data
m = Ridge(fit_intercept=False, alpha=alpha_opt)
X_train = generate_design_matrix(x_train, y_train, n=p_opt)
X_test = generate_design_matrix(x_test, y_test, n=p_opt)

m.fit(X_train, values_train)
zhat_train = m.predict(X_train)
zhat_test = m.predict(X_test)


r2_train, r2_test = r2_score(values_train, zhat_train), r2_score(values_test, zhat_test)
print("train", r2_train, "\ttest", r2_test)

plt.show()
