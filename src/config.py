# src/config.py

import numpy as np

# ---- DGP ----
DGP_DEFAULTS = dict(T_train=500, T_test=2000, c=10.0, b_star=1.0)

# ---- Grids ----
CQ_GRID = [0.5, 0.75, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 7, 10]
Z_GRID  = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]

# ---- Task 2 Lasso ----
LASSO_ALPHA_FRACTIONS = np.geomspace(1.0, 0.005, 16)
LASSO_VALIDATION_FRACTION = 0.2
LASSO_TOL = 1e-4
LASSO_PATH_MAX_ITER = 10000
LASSO_REFIT_MAX_ITER = 20000
LASSO_FIT_INTERCEPT = False
LASSO_SELECTION = "cyclic"

# ---- Task 3 validation ----
TASK3_OUTER_TRAIN_FRACTIONS = (0.6, 0.7, 0.8)
TASK3_INNER_VALIDATION_FRACTION = 0.2
TASK3_RIDGE_Z_GRID = Z_GRID + [200, 500, 1000, 5000, 10000]
TASK3_ELASTIC_L1_RATIOS = (0.1, 0.5, 0.9)
TASK3_ELASTIC_ALPHA_FRACTIONS = np.geomspace(1.0, 0.01, 12)
TASK3_COMPONENT_COUNTS = (1, 2, 3, 5, 10, 20, 40, 60, 80)
TASK3_ELASTIC_MAX_ITER = 20000
TASK3_ELASTIC_TOL = 1e-4
TASK3_PLS_MAX_ITER = 500
TASK3_PLS_TOL = 1e-6

# ---- Reproducibility ----
N_RUNS = 20
SEED = 0

# ---- Paths ----
FIG_DIR    = "figures"
RESULT_DIR = "results"
