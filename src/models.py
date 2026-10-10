# src/models.py

import numpy as np
import pandas as pd
import warnings

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, lasso_path
from sklearn.pipeline import make_pipeline
from .config import (
    LASSO_ALPHA_FRACTIONS,
    LASSO_FIT_INTERCEPT,
    LASSO_PATH_MAX_ITER,
    LASSO_REFIT_MAX_ITER,
    LASSO_SELECTION,
    LASSO_TOL,
    LASSO_VALIDATION_FRACTION,
    TASK3_COMPONENT_COUNTS,
    TASK3_ELASTIC_ALPHA_FRACTIONS,
    TASK3_ELASTIC_L1_RATIOS,
    TASK3_ELASTIC_MAX_ITER,
    TASK3_ELASTIC_TOL,
    TASK3_INNER_VALIDATION_FRACTION,
    TASK3_PLS_MAX_ITER,
    TASK3_PLS_TOL,
    TASK3_RIDGE_Z_GRID,
)


def get_torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_two_layer_net(input_dim, width, device):
    import torch
    from torch import nn

    class TwoLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(input_dim, width)
            self.output = nn.Linear(width, 1)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            return self.output(x).squeeze(-1)

    return TwoLayerNet().to(device)


def evaluate_neural_network(model, X, R, device):
    import torch

    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    R_tensor = torch.as_tensor(R, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(X_tensor)
        mse = torch.mean((R_tensor - pred) ** 2).item()
        r2 = 1.0 - mse / torch.mean(R_tensor ** 2).item()

        timing_return = pred * R_tensor
        expected_return = torch.mean(timing_return).item()
        second_moment = torch.mean(timing_return ** 2).item()
        sharpe = (
            expected_return / np.sqrt(second_moment)
            if second_moment > 1e-12 else 0.0
        )

    return {
        "mse": mse,
        "r2": r2,
        "expected_return": expected_return,
        "sharpe": sharpe,
    }


def neural_network_complexity(model):
    return model.output.weight.detach().abs().sum().item()


def train_two_layer_network(
    model,
    X_train,
    R_train,
    X_test,
    R_test,
    epochs,
    batch_size,
    lr,
    device,
    seed=0,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seed_torch(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(R_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_neural_network(model, X_train, R_train, device)
        test_metrics = evaluate_neural_network(model, X_test, R_test, device)
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_metrics["mse"],
            "test_mse": test_metrics["mse"],
            "test_r2": test_metrics["r2"],
            "test_return": test_metrics["expected_return"],
            "test_sharpe": test_metrics["sharpe"],
            "complexity": neural_network_complexity(model),
        })

    return pd.DataFrame(history)


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


def fit_ridge_with_validation(
    X_train,
    y_train,
    z_values,
    validation_fraction=TASK3_INNER_VALIDATION_FRACTION,
):
    """Select a Ridge penalty on a chronological holdout, then refit all rows."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2 or y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
        raise ValueError("X_train must be 2D and y_train must be a matching 1D array")
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        raise ValueError("Training data must contain only finite values")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    n_train = X_train.shape[0]
    n_fit = int((1.0 - validation_fraction) * n_train)
    if n_fit <= 0 or n_fit >= n_train:
        raise ValueError("Training data is too small for the configured validation split")

    z_values = np.atleast_1d(np.asarray(z_values, dtype=float))
    X_fit, y_fit = X_train[:n_fit], y_train[:n_fit]
    X_val, y_val = X_train[n_fit:], y_train[n_fit:]
    candidate_betas = fit_ridge_path(X_fit, y_fit, z_values)
    validation_predictions = X_val @ candidate_betas.T
    validation_mses = np.mean((y_val[:, None] - validation_predictions) ** 2, axis=0)
    if not np.all(np.isfinite(validation_mses)):
        raise RuntimeError("Ridge path produced non-finite validation MSE values")

    best_index = int(np.argmin(validation_mses))
    selected_z = float(z_values[best_index])
    selected_validation_mse = float(validation_mses[best_index])
    final_beta = fit_ridge(X_train, y_train, selected_z)
    return final_beta, selected_z, selected_validation_mse


def _build_task3_estimator(model_name, parameters, n_fit):
    if model_name == "ridge":
        return Ridge(alpha=parameters["z"] * n_fit, fit_intercept=False, solver="svd")
    if model_name == "ridge_intercept":
        return Ridge(alpha=parameters["z"] * n_fit, fit_intercept=True, solver="svd")
    if model_name == "lasso":
        return Lasso(
            alpha=parameters["alpha"],
            fit_intercept=False,
            max_iter=LASSO_REFIT_MAX_ITER,
            tol=LASSO_TOL,
            selection=LASSO_SELECTION,
        )
    if model_name == "elastic_net":
        return ElasticNet(
            alpha=parameters["alpha"],
            l1_ratio=parameters["l1_ratio"],
            fit_intercept=True,
            max_iter=TASK3_ELASTIC_MAX_ITER,
            tol=TASK3_ELASTIC_TOL,
            selection="cyclic",
        )
    if model_name == "pcr":
        return make_pipeline(
            PCA(n_components=parameters["n_components"], svd_solver="full"),
            LinearRegression(),
        )
    if model_name == "pls":
        return PLSRegression(
            n_components=parameters["n_components"],
            scale=True,
            max_iter=TASK3_PLS_MAX_ITER,
            tol=TASK3_PLS_TOL,
        )
    raise ValueError(f"Unsupported Task 3 model: {model_name}")


def fit_task3_model_with_validation(
    model_name,
    X_train,
    y_train,
    validation_fraction=TASK3_INNER_VALIDATION_FRACTION,
    ridge_z_grid=None,
    alpha_fractions=None,
    elastic_l1_ratios=None,
    component_counts=None,
):
    """Tune a Task 3 model on a chronological holdout and refit all rows."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if X_train.ndim != 2 or y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
        raise ValueError("X_train must be 2D and y_train must be a matching 1D array")
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        raise ValueError("Training data must contain only finite values")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    n_train = X_train.shape[0]
    n_fit = int((1.0 - validation_fraction) * n_train)
    if n_fit <= 1 or n_fit >= n_train:
        raise ValueError("Training data is too small for the configured validation split")

    if ridge_z_grid is None:
        ridge_z_grid = TASK3_RIDGE_Z_GRID
    if alpha_fractions is None:
        alpha_fractions = LASSO_ALPHA_FRACTIONS
    if elastic_l1_ratios is None:
        elastic_l1_ratios = TASK3_ELASTIC_L1_RATIOS
    if component_counts is None:
        component_counts = TASK3_COMPONENT_COUNTS

    X_fit, y_fit = X_train[:n_fit], y_train[:n_fit]
    X_valid, y_valid = X_train[n_fit:], y_train[n_fit:]
    settings = []
    if model_name in {"ridge", "ridge_intercept"}:
        settings = [{"z": float(z)} for z in ridge_z_grid]
    elif model_name == "lasso":
        alpha_max = np.max(np.abs(X_fit.T @ y_fit)) / n_fit
        if not np.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError("Could not construct a positive Lasso alpha grid")
        settings = [{"alpha": float(alpha_max * fraction)} for fraction in alpha_fractions]
    elif model_name == "elastic_net":
        X_centered = X_fit - X_fit.mean(axis=0)
        y_centered = y_fit - y_fit.mean()
        max_correlation = np.max(np.abs(X_centered.T @ y_centered)) / n_fit
        if not np.isfinite(max_correlation) or max_correlation <= 0:
            raise ValueError("Could not construct a positive Elastic Net alpha grid")
        settings = [
            {"alpha": float(max_correlation * fraction / l1_ratio), "l1_ratio": float(l1_ratio)}
            for l1_ratio in elastic_l1_ratios
            for fraction in TASK3_ELASTIC_ALPHA_FRACTIONS
        ]
    elif model_name in {"pcr", "pls"}:
        max_components = min(n_fit - 1, X_fit.shape[1])
        settings = [
            {"n_components": int(count)}
            for count in component_counts
            if 1 <= count <= max_components
        ]
    else:
        raise ValueError(f"Unsupported Task 3 model: {model_name}")
    if not settings:
        raise ValueError(f"No valid hyperparameter candidates for {model_name}")

    best_mse = np.inf
    best_parameters = None
    for parameters in settings:
        estimator = _build_task3_estimator(model_name, parameters, n_fit)
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(X_fit, y_fit)
        if any(issubclass(item.category, ConvergenceWarning) for item in fit_warnings):
            continue
        if hasattr(estimator, "dual_gap_"):
            tolerance = TASK3_ELASTIC_TOL if model_name == "elastic_net" else LASSO_TOL
            _check_lasso_convergence(estimator.dual_gap_, y_fit, "Task 3 tuning", tolerance)
        predictions = np.asarray(estimator.predict(X_valid)).reshape(-1)
        if not np.all(np.isfinite(predictions)):
            continue
        validation_mse = float(np.mean((y_valid - predictions) ** 2))
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_parameters = parameters

    if best_parameters is None:
        raise RuntimeError(f"No converged finite candidate found for {model_name}")

    final_estimator = _build_task3_estimator(model_name, best_parameters, n_train)
    with warnings.catch_warnings(record=True) as refit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        final_estimator.fit(X_train, y_train)
    if any(issubclass(item.category, ConvergenceWarning) for item in refit_warnings):
        raise RuntimeError(f"Final {model_name} refit did not converge")
    if hasattr(final_estimator, "dual_gap_"):
        tolerance = TASK3_ELASTIC_TOL if model_name == "elastic_net" else LASSO_TOL
        _check_lasso_convergence(final_estimator.dual_gap_, y_train, "Task 3 refit", tolerance)

    metadata = {
        **best_parameters,
        "inner_validation_mse": best_mse,
        "n_candidates": len(settings),
    }
    return final_estimator, metadata


def _check_lasso_convergence(dual_gap, y, stage, tol):
    dual_gap = np.asarray(dual_gap, dtype=float)
    stopping_threshold = tol * np.mean(np.asarray(y, dtype=float) ** 2)
    if not np.all(np.isfinite(dual_gap)) or np.any(dual_gap > stopping_threshold):
        raise RuntimeError(f"Lasso {stage} did not meet the dual-gap convergence tolerance")


def fit_lasso_with_validation(
    X_train,
    y_train,
    alpha_fractions=None,
    validation_fraction=LASSO_VALIDATION_FRACTION,
    tol=LASSO_TOL,
    max_iter=LASSO_PATH_MAX_ITER,
    refit_max_iter=LASSO_REFIT_MAX_ITER,
):
    """Select Lasso alpha on a held-out training block, then refit all rows."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2 or y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
        raise ValueError("X_train must be 2D and y_train must be a matching 1D array")
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        raise ValueError("Training data must contain only finite values")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    if alpha_fractions is None:
        alpha_fractions = LASSO_ALPHA_FRACTIONS
    alpha_fractions = np.asarray(alpha_fractions, dtype=float)
    if (
        alpha_fractions.ndim != 1
        or alpha_fractions.size == 0
        or not np.all(np.isfinite(alpha_fractions))
        or np.any(alpha_fractions <= 0)
        or np.any(alpha_fractions > 1)
        or np.any(np.diff(alpha_fractions) >= 0)
    ):
        raise ValueError("alpha_fractions must be finite, positive, decreasing fractions no greater than 1")

    n_train = X_train.shape[0]
    n_fit = int((1.0 - validation_fraction) * n_train)
    if n_fit <= 0 or n_fit >= n_train:
        raise ValueError("Training data is too small for the configured validation split")

    X_fit, y_fit = X_train[:n_fit], y_train[:n_fit]
    X_val, y_val = X_train[n_fit:], y_train[n_fit:]
    alpha_max = np.max(np.abs(X_fit.T @ y_fit)) / n_fit
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        raise ValueError("Could not construct a positive Lasso alpha grid")

    alpha_grid = alpha_max * alpha_fractions
    with warnings.catch_warnings(record=True) as path_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        alphas, coefficients, dual_gaps = lasso_path(
            X_fit,
            y_fit,
            alphas=alpha_grid,
            max_iter=max_iter,
            tol=tol,
        )
    if any(issubclass(item.category, ConvergenceWarning) for item in path_warnings):
        raise RuntimeError("Lasso path did not converge for one or more candidate alphas")
    _check_lasso_convergence(dual_gaps, y_fit, "path", tol)
    if not np.all(np.isfinite(alphas)) or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("Lasso path returned non-finite alphas or coefficients")

    validation_predictions = X_val @ coefficients
    validation_mses = np.mean((y_val[:, None] - validation_predictions) ** 2, axis=0)
    if not np.all(np.isfinite(validation_mses)):
        raise RuntimeError("Lasso path produced non-finite validation MSE values")
    best_index = int(np.argmin(validation_mses))
    selected_alpha = float(alphas[best_index])
    selected_alpha_validation_mse = float(validation_mses[best_index])

    final_model = Lasso(
        alpha=selected_alpha,
        fit_intercept=LASSO_FIT_INTERCEPT,
        max_iter=refit_max_iter,
        tol=tol,
        selection=LASSO_SELECTION,
    )
    with warnings.catch_warnings(record=True) as refit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model.fit(X_train, y_train)
    if any(issubclass(item.category, ConvergenceWarning) for item in refit_warnings):
        raise RuntimeError("Final Lasso refit did not converge")
    _check_lasso_convergence(final_model.dual_gap_, y_train, "refit", tol)
    if not np.all(np.isfinite(final_model.coef_)):
        raise RuntimeError("Final Lasso refit returned non-finite coefficients")

    return final_model.coef_, selected_alpha, selected_alpha_validation_mse
