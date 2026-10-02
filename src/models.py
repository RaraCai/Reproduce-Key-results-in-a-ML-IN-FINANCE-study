# src/models.py

import numpy as np

def fit_ridge(X, y, z, rcond=1e-12):
    """
    Ridge with no intercept, matching KMZ:
        beta_hat(z) = (X'X + z*T_tr*I)^{-1} X'y
    z = 0 -> minimum-norm least squares (ridgeless).
    """
    T_train, P1 = X.shape

    if z <= 0:
        # ridgeless: minimum-norm LS via SVD/lstsq
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=rcond)
        return beta_hat

    # augmented LS form: numerically stable, avoids forming X'X
    X_aug = np.vstack([X, np.sqrt(z * T_train) * np.eye(P1)])
    y_aug = np.concatenate([y, np.zeros(P1)])
    beta_hat, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return beta_hat
