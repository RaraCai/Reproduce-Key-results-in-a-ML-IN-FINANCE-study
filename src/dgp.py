# src/dgp.py

import numpy as np

def generate_nonlinear_single_index_data(T, D, U, noise_std=0.5, seed=0):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(T, D))
    latent = X @ U
    signal = latent + latent**3 / 6.0
    noise = rng.normal(0.0, noise_std, size=T)
    R = signal + noise

    return X.astype(np.float32), R.astype(np.float32)


def generate_dgp_data(
    T_train = 500,
    T_test = 2000,
    c = 10.0,
    b_star = 1.0,
    seed = None
):
    """
    Generates synthetic data from a dense linear DGP(Data Generating Process): 
    R = S @ beta_star + epsilon.
    """
    rng = np.random.default_rng(seed)

    P = int(c * T_train)    # P features
    T_total = T_train + T_test

    # 1. generate feature matrix S
    S = rng.standard_normal((T_total, P))
    # 2. generate Dense β* then normalize it
    beta_raw = rng.standard_normal(P)
    beta_star = beta_raw * np.sqrt(b_star / np.sum(beta_raw **2))
    # 3. generate noise
    epsilon = rng.standard_normal(T_total)
    # 4. generate R_{t+1} = S_t' * β* + epsilon_{t+1}
    R = S @ beta_star + epsilon

    # split data into training set and testing set
    S_train, R_train = S[:T_train], R[:T_train]
    S_test, R_test = S[T_train:], R[T_train:]

    return S_train, R_train, S_test, R_test, beta_star, P


def generate_dgp_run(seed=None, **dgp_parameters):
    """Generate one reproducible run, including Task 1's column permutation."""
    S_train, R_train, S_test, R_test, beta_star, P = generate_dgp_data(
        **dgp_parameters,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(P)
    return S_train, R_train, S_test, R_test, beta_star, P, permuted_indices


def get_observed_block(
    S_train,
    S_test,
    P1,
    permuted_indices
):
    """
    Simulate the misspecified settings and create the partially observed environment
    """
    P = S_train.shape[1]
    P1 = int(min(P, max(1, P1)))    # observable features

    observed_cols = permuted_indices[:P1]   # apply a random permutation to columns of S
    X_train = S_train[:, observed_cols]
    X_test = S_test[:, observed_cols]

    return X_train, X_test