# src/metrics.py

import numpy as np

def eval_oos_performance(
    beta_hat,
    X_test,
    R_test
):
    """
    OOS metrics matching KMZ:
      R2_paper   = 1 - E[(R - R_hat)^2] / E[R^2]      (Prop. 3, uncentered)
      exp_return = E[R^pi],  R^pi = R_hat * R          (timing return)
      sharpe     = E[R^pi] / sqrt(E[(R^pi)^2])         (Eq. 5, uncentered)
      sharpe_var = E[R^pi] / sqrt(Var(R^pi))           (secondary diagnostic)
      beta_norm_sq = ||beta_hat||_2^2

    Returns a dict of scalars.
    """
    R_hat = X_test @ beta_hat
    R_pi  = R_hat * R_test

    # Uncentered R^2: denominator uses E[R^2], not Var(R)
    mse = np.mean((R_test - R_hat) ** 2)
    r2_paper = 1.0 - mse / np.mean(R_test ** 2)

    exp_return = np.mean(R_pi)

    # Uncentered second moment in the denominator, per KMZ Eq. (5)
    second_moment = np.mean(R_pi ** 2)
    sharpe = exp_return / np.sqrt(second_moment) if second_moment > 1e-12 else 0.0

    # Centered version as a secondary diagnostic
    var_pi = np.var(R_pi)
    sharpe_var = exp_return / np.sqrt(var_pi) if var_pi > 1e-12 else 0.0

    beta_norm_sq = float(np.sum(beta_hat ** 2))

    return dict(
        r2_paper=r2_paper,
        exp_return=exp_return,
        sharpe=sharpe,
        sharpe_var=sharpe_var,
        beta_norm_sq=beta_norm_sq,
    )


def expand_beta(beta_hat, observed_cols, P):
    """Map P1-dim beta_hat back to full P-dim space (unobserved coords = 0)."""
    full = np.zeros(P)
    full[observed_cols] = beta_hat
    return full


def param_error(beta_hat, observed_cols, P, beta_star):
    """||beta_hat_full - beta_star||_2, only meaningful in synthetic data."""
    return float(np.linalg.norm(expand_beta(beta_hat, observed_cols, P) - beta_star))