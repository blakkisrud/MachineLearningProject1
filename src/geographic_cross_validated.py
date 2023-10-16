from utils import *

import pandas as pd
import numpy as np


OUTPUT_DIR = "output_terrain"
CALCULATE_SCORES = True    # Set to false to load pre-calculated scores
SPLIT_ON_CHUNK = True      # True: split image in train / test CHUNKS, else by random pixels
MODEL_FUNC = "lasso"

# avals = np.logspace(0, 5, 12)   # regularization strengths
avals = np.logspace(-6, 1, 8)
pvals = list(range(5))                         # max polynomial degrees
k_val_split = 3                                 # number of folds in KFold validation split for HP-tuning
print(avals)


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# TODO: average across geographic data - model generalization?

gen = get_terrain_data()
# next(gen)
# next(gen)

for terrain, TERRAIN_NAME in gen:
    print("\n", TERRAIN_NAME, terrain.shape)

    ta = TerrainAnalyser(terrain=terrain, terrain_name=TERRAIN_NAME, avals=avals, pvals=pvals, k_val_split=k_val_split,
                         calculate_scores=CALCULATE_SCORES, split_on_chunk=SPLIT_ON_CHUNK, output_dir=OUTPUT_DIR, model_func=MODEL_FUNC)

    ta.show_terrain_partitions(showplt=False)


    if CALCULATE_SCORES:
        ta.calculate_k_fold_cross_validated()

    ta.load_k_fold_cross_validated()
    ta.show_k_fold_cross_validated_scores(showplt=False)


    ta.find_optimal_hps()
    ta.evaluate_on_test()
    break

plt.show()