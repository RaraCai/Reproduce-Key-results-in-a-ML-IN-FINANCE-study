# src/config.py

# ---- DGP ----
DGP_DEFAULTS = dict(T_train=500, T_test=2000, c=10.0, b_star=1.0)

# ---- Grids ----
CQ_GRID = [0.5, 0.75, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 7, 10]
Z_GRID  = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]

# ---- Reproducibility ----
SEED = 0

# ---- Paths ----
FIG_DIR    = "figures"
RESULT_DIR = "results"

# ---- Plotting ----
DPI        = 300
FIG_FORMAT = "pdf"