from utils import *
# from analysis_franke import *
import pandas as pd

OUTPUT_DIR = "output_terrain"
STEP_SIZE = 0.01
SIGMA = 0.1


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Getting ONE terrain + Franke function
# Splitting into train / test TWICE: chunk mode and random pixel mode

terrain_gen = get_terrain_data()
# data_terrain, name_terrain = next(terrain_gen) # do next until you get the image you want

# tr_chunk = TerrainAnalyser(terrain=data_terrain, terrain_name="Terrain chunk split", calculate_scores=False, split_on_chunk=True, output_dir=OUTPUT_DIR)
# tr_pix = TerrainAnalyser(terrain=data_terrain, terrain_name="Terrain random split", calculate_scores=False, split_on_chunk=False, output_dir=OUTPUT_DIR)


# x, y = np.meshgrid(np.arange(0, 1, STEP_SIZE), np.arange(0, 1, STEP_SIZE))
# data_franke = FrankeFunction(x, y, add_noise=True)

# fr_chunk = TerrainAnalyser(terrain=data_franke, terrain_name="Franke chunk split", split_on_chunk=True, calculate_scores=False, output_dir=OUTPUT_DIR)
# fr_pix = TerrainAnalyser(terrain=data_franke, terrain_name="Franke random split", split_on_chunk=False, calculate_scores=False, output_dir=OUTPUT_DIR)


# FIGURE METHODS - NORMALIZATION
# PLOT IMAGE DATA DISTRIBUTIONS FOR BOTH SPLIT MODES, BOTH BEFORE AND AFTER NORMALIZING
# DECISION IS MADE TO USE STANDARD-SCORE NORMALIZATION, ONLY BASED ON TEST-DATA

dims = (12, 8)
fig1, ax1 = plt.subplots(ncols=2, nrows=2, figsize=dims)
# fig1.figsize = dims
ax1 = ax1.ravel()

fig2, ax2 = plt.subplots(ncols=2, nrows=2)
ax2 = ax2.ravel()

fig3, ax3 = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
ax3 = ax3.ravel()

fig1.suptitle("Original values")
fig2.suptitle("Post-normalization")


# for i, ta in enumerate([tr_chunk, tr_pix, fr_chunk, fr_pix]):
for i, data in enumerate(terrain_gen):
    terrain, terrain_name = data
    tr_chunk = TerrainAnalyser(terrain=terrain, terrain_name=f"{terrain_name} chunk split", calculate_scores=False, split_on_chunk=True, output_dir=OUTPUT_DIR)
    tr_pix = TerrainAnalyser(terrain=terrain, terrain_name=f"{terrain_name} random split", calculate_scores=False, split_on_chunk=False, output_dir=OUTPUT_DIR)

    # print(ta.terrain_name)

    # PLOTTING DISTRIBUTIONS
    # ax1[i].hist(ta.values_train_orig.ravel())
    # ax2[i].hist(ta.values_train)

    # ax1[i].set_title(ta.terrain_name)
    # ax2[i].set_title(ta.terrain_name)


    # R2 and MSE for increasing model complexity
    # OLS

    pvals = []
    r2_train_vals = []
    mse_train_vals = []
    r2_test_vals = []
    mse_test_vals = []

    for ta in [tr_chunk, tr_pix]:
        print(ta.terrain_name)

        # ta.ols(pmax=10)
        # scores = ta.load_ols()
        ta.load_k_fold_cross_validated()

        # pvals = scores.index.values

        # sns.scatterplot(data=scores)
        # ax3[0].plot(pvals, scores["r2 train"], "o", label=ta.terrain_name)
        # ax3[1].plot(pvals, scores["mse train"], "o", label=ta.terrain_name)
        # ax3[2].plot(pvals, scores["r2 test"], "o", label=tr_chunk.terrain_name)
        # ax3[3].plot(pvals, scores["mse test"], "o", label=ta.terrain_name)


ax3[0].set_ylabel("R2 train")
ax3[1].set_ylabel("MSE train")
ax3[2].set_ylabel("R2 test")
ax3[3].set_ylabel("MSE test")
[axx.set_xlabel("p") for axx in ax3]

ax3[0].legend()
fig3.savefig(os.path.join(OUTPUT_DIR, "terrain_ols.pdf"))
plt.show()


fig1.savefig(os.path.join(OUTPUT_DIR, "distibutions.pdf"))

# plt.show()
