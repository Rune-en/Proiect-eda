"""
Application configuration – 12-factor Factor III.

All tuneable values are read from environment variables so that the same
codebase can be promoted across environments (dev → staging → prod) without
code changes.  A `.env` file (git-ignored) can supply defaults locally;
copy `.env.example` to `.env` and adjust as needed.
"""

import os

from dotenv import load_dotenv

# Load variables from a local .env file when present (dev convenience).
# In production, real environment variables take precedence automatically.
load_dotenv()

# ---------------------------------------------------------------------------
# Data paths (Factor IV – Backing Services treated as attached resources)
# ---------------------------------------------------------------------------

# Absolute or relative path to the raw gene-expression CSV file.
DATASET_PATH: str = os.getenv("DATASET_PATH", "DB.csv")

# Destination path for the preprocessed / transposed CSV written by the pipeline.
OUTPUT_PATH: str = os.getenv("OUTPUT_PATH", "processed_data.csv")

# Directory where per-model result CSVs are written (one file per model).
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")
# Directory where per-reduction CSVs are saved (train + test for each method).
REDUCTIONS_DIR: str = os.getenv("REDUCTIONS_DIR", "results/reductions")
# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Column name used as the prediction target after transposing the dataset.
TARGET_COLUMN: str = os.getenv("TARGET_COLUMN", "Gleason Group")

# ---------------------------------------------------------------------------
# Model / experiment settings
# ---------------------------------------------------------------------------

# Proportion of samples reserved for the test split.
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))

# Global random seed – ensures reproducible train/test splits and model fits.
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))

# ---------------------------------------------------------------------------
# Pipeline feature flags
# ---------------------------------------------------------------------------

# When True, use 5-fold stratified cross-validation; when False, use a single
# stratified train/test split (TEST_SIZE fraction held out).
USE_CV: bool = os.getenv("USE_CV", "true").lower() in ("true", "1", "yes")

# When True, apply SMOTE oversampling to each reduced training fold.
USE_SMOTE: bool = os.getenv("USE_SMOTE", "true").lower() in ("true", "1", "yes")

# When True, remove near-zero-variance features from the full feature matrix
# before any fold-level processing (global preprocessing step).
REMOVE_LOW_VARIANCE: bool = os.getenv("REMOVE_LOW_VARIANCE", "false").lower() in ("true", "1", "yes")

# Variance threshold used when REMOVE_LOW_VARIANCE is True.
# Features with variance < this value (in log2 space) are dropped.
VARIANCE_THRESHOLD: float = float(os.getenv("VARIANCE_THRESHOLD", "0.1"))
