# DSA5205 Project 1

## Project Summary

### File Structure

```text
.
├── requirements.txt                 # Python dependencies
├── datasets/                        # Task 3 training and public-test data
├── notebooks/
│   ├── task1.ipynb                  # Ridge simulation and benchmark
│   ├── task2.ipynb                  # Lasso validation and Ridge comparison
│   ├── task3.ipynb                  # Model validation and prediction export
│   └── task4.ipynb                  # Two-layer neural-network experiments
├── src/
│   ├── config.py                    # Shared parameters and output paths
│   ├── dgp.py                       # Data-generating process
│   ├── metrics.py                   # Evaluation metrics
│   ├── models.py                    # Model fitting and validation routines
│   └── plotting.py                  # Plotting helpers
├── figures/                         # Generated figures
├── results/                         # Generated metrics and summaries
└── predictions/                     # Task 3 prediction CSV files
```

### Related Links

[See the code repository in Github.](https://github.com/RaraCai/Reproduce-Key-results-in-a-ML-IN-FINANCE-study.git)

[See the project video in Youtube.](https://youtu.be/5vBW-ltxXiI)



## Requirements

Python 3.10 or later is recommended. From the project root, create and activate a virtual environment, then install all dependencies listed in `requirements.txt`:

```bash
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
```

The requirements file installs NumPy, SciPy, pandas, Matplotlib, tqdm, scikit-learn, PyTorch, and Jupyter.

## Running the Notebooks

Run the notebooks from the project root, in the order shown below. For each command, Jupyter opens the notebook in a browser; select **Run > Run All Cells** and wait for execution to finish before moving to the next notebook.

```bash
    python -m jupyter notebook notebooks/task1.ipynb
    python -m jupyter notebook notebooks/task2.ipynb
    python -m jupyter notebook notebooks/task3.ipynb
    python -m jupyter notebook notebooks/task4.ipynb
```

Task 2 uses the Ridge benchmark produced by Task 1, so Task 1 must finish successfully first. Task 3 reads the supplied files in `datasets`. Task 4 selects CUDA when available and otherwise runs on CPU.

## Expected Outputs

After running the notebooks, the following files are created or updated:

- **Task1**:
  - `task1_ridge_metrics.npz`
  - `task1_r2_paper.pdf`
  - `task1_expected_return.pdf`
  - `task1_sharpe_ratio.pdf`
  - `task1_beta_norm.pdf`


- **Task2**:
  - `task2_lasso_metrics.npz`
  - `task2_lasso_summary.csv`
  - `task2_lasso_vs_ridge_beta_norm_sq.pdf`
  - `task2_lasso_vs_ridge_exp_return.pdf`
  - `task2_lasso_vs_ridge_r2_paper.pdf`
  - `task2_lasso_vs_ridge_sharpe.pdf`


- **Task3**:
    -`predictions/<STUDENT_ID>_predictions_A.csv`
    - `predictions/<STUDENT_ID>_predictions_B.csv`
    - `predictions/<STUDENT_ID>_predictions_C.csv`
    
    Each prediction CSV contains `t` and `yhat` columns, with rows in public-test order.


- **Task4**:
  - `task4_training_dynamics.csv`
  - `task4_stopping_comparison.csv`
  - `task4_train_test_mse.pdf`
  - `task4_oos_r2.pdf`
  - `task4_oos_sharpe.pdf`
  - `task4_network_complexity.pdf`

The notebooks also display validation metrics, summaries, and plots during execution. Output directories are created by the notebooks when needed.