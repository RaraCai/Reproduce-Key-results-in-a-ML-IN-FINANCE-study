# src/models.py

import numpy as np

def fit_ridge_path(X, y, z_values, rcond=1e-12):
    """
    Fit ridge models for multiple penalties using one SVD of X.

    For z > 0, beta_hat(z) = V diag(s / (s^2 + z*T_train)) U' y.
    For z = 0, use the minimum-norm pseudoinverse with cutoff rcond.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    z_values = np.atleast_1d(np.asarray(z_values, dtype=float))

    if X.ndim != 2 or y.ndim not in (1, 2) or y.shape[0] != X.shape[0]:
        raise ValueError("X must be 2D and y must have a matching leading dimension")
    if not np.all(np.isfinite(z_values)) or np.any(z_values < 0):
        raise ValueError("z_values must contain finite, non-negative penalties")

    n_train = X.shape[0]
    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    projected_targets = U.T @ y

    shrinkage = np.empty((len(z_values), len(singular_values)))
    for index, z in enumerate(z_values):
        if z == 0:
            cutoff = singular_values[0] * rcond if singular_values.size else 0.0
            shrinkage[index] = np.divide(
                1.0,
                singular_values,
                out=np.zeros_like(singular_values),
                where=singular_values > cutoff,
            )
        else:
            shrinkage[index] = singular_values / (singular_values ** 2 + z * n_train)

    if y.ndim == 1:
        spectral_coefficients = shrinkage * projected_targets[None, :]
    else:
        spectral_coefficients = shrinkage[:, :, None] * projected_targets[None, :, :]
        return np.einsum("krt,rp->kpt", spectral_coefficients, Vt)
    return spectral_coefficients @ Vt


def fit_ridge(X, y, z, rcond=1e-12):
    """Fit one no-intercept ridge model; z = 0 gives the minimum-norm solution."""
    return fit_ridge_path(X, y, [z], rcond=rcond)[0]
