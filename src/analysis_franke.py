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

# Global plotting parameters

sns.set_style("whitegrid")
sns.set_context("paper")

# Global variables

SIGMA = 0.1 # Noise level
OUTPUT_DIR = "output_tmp" # Output directory for figures
STEP_SIZE = 0.01 # Step size for sampling the Franke function

# Running flags

DO_PART_A = False
DO_PART_B = False
DO_PART_C = False
DO_PART_D = False
DO_PART_E = False
DO_PART_F = False

# Helper functions

# Functions for doing the analysis, one function per subtask

def part_a():
    print("Doing part a")

# Run the parts

if DO_PART_A:
    part_a()


