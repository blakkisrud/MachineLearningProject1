from utils import *
from imageio.v3 import imread

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import resample


terrain = imread("SRTM_data_Norway_1.tif")
# terrain = imread("SRTM_data_Norway_2.tif")

# plt.figure()
# plt.imshow(terrain)
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

# TODO: nest into cross-validation?
scaler_img = StandardScaler()
values_train = scaler_img.fit_transform(terrain_values[idx_train].reshape(-1, 1))
values_test = scaler_img.transform(terrain_values[idx_test].reshape(-1, 1))

scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler_x.fit_transform(x_train).reshape(-1)
x_test = scaler_x.transform(x_test).reshape(-1)
y_train = scaler_y.fit_transform(y_train).reshape(-1)
y_test = scaler_y.transform(y_test).reshape(-1)


def show_train_test_partitions():
    img_train = np.empty(shape=len(index_vals))
    img_test = np.empty(shape=len(index_vals))

    img_train[idx_train] = values_train.ravel()
    img_test[idx_test] = values_test.ravel()

    img_train = img_train.reshape(terrain.shape)
    img_test = img_test.reshape(terrain.shape)

    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(img_train)
    ax[0].set_title("Train")
    ax[1].imshow(img_test)
    ax[1].set_title("Test")
    ax[2].imshow(img_train + img_test)
    ax[2].set_title("Train + test")
    [axi.axis("off") for axi in ax]

    plt.show()
    pass


# show_train_test_partitions()


# Divide train set into K folds for hyperparameter-tuning using cross-validation
print(x_train.shape, y_train.shape)
