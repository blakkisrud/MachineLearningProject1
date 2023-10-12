from utils import *
# from analysis_franke import *

OUTPUT_DIR = "output_methods"
STEP_SIZE = 0.01
SIGMA = 0.1


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Getting ONE terrain + Franke function
# Splitting into train / test TWICE: chunk mode and random pixel mode

terrain_gen = get_terrain_data()
data_terrain, name_terrain = next(terrain_gen) # do next until you get the image you want

tr_chunk = TerrainAnalyser(terrain=data_terrain, terrain_name="terrain chunk", calculate_scores=False, split_on_chunk=True, output_dir=OUTPUT_DIR)

tr_pix = TerrainAnalyser(terrain=data_terrain, terrain_name="terrain random", calculate_scores=False, split_on_chunk=False, output_dir=OUTPUT_DIR)


x, y = np.meshgrid(np.arange(0, 1, STEP_SIZE), np.arange(0, 1, STEP_SIZE))
data_franke = FrankeFunction(x, y, add_noise=True)

fr_chunk = TerrainAnalyser(terrain=data_franke, terrain_name="Franke chunk", split_on_chunk=True, calculate_scores=False, output_dir=OUTPUT_DIR)
fr_pix = TerrainAnalyser(terrain=data_franke, terrain_name="Franke random", split_on_chunk=False, calculate_scores=False, output_dir=OUTPUT_DIR)


# FIGURE METHODS - NORMALIZATION
# PLOT IMAGE DATA DISTRIBUTIONS FOR BOTH SPLIT MODES, BOTH BEFORE AND AFTER NORMALIZING
# DECISION IS MADE TO USE STANDARD-SCORE NORMALIZATION, ONLY BASED ON TEST-DATA

fig1, ax1 = plt.subplots(ncols=2, nrows=2)
ax1 = ax1.ravel()

fig2, ax2 = plt.subplots(ncols=2, nrows=2)
ax2 = ax2.ravel()

fig1.suptitle("Original values")
fig2.suptitle("Post-normalization")

for i, ta in enumerate([tr_chunk, tr_pix, fr_chunk, fr_pix]):
    print(ta.terrain_name)
    ta.show_terrain_partitions()    # uncomment to remove plots of all train / test partitions of the images (four in total)

    ax1[i].hist(ta.values_train_orig.ravel())
    ax2[i].hist(ta.values_train)

    ax1[i].set_title(ta.terrain_name)
    ax2[i].set_title(ta.terrain_name)


plt.show()
