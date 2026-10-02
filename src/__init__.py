# src/__init__.py

from .config import DGP_DEFAULTS, CQ_GRID, Z_GRID, SEED, FIG_DIR, RESULT_DIR
from .dgp import generate_dgp_data, get_observed_block
from .models import fit_ridge
from .metrics import eval_oos_performance, expand_beta, param_error

__all__ = [
    # Config
    "DGP_DEFAULTS",
    "CQ_GRID",
    "Z_GRID",
    "SEED",
    "FIG_DIR",
    "RESULT_DIR",
    # DGP & Model & Metrics
    "generate_dgp_data",
    "get_observed_block",
    "fit_ridge",
    "eval_oos_performance",
    "expand_beta",
    "param_error",
]