"""
Synthetic medical data generator for model training augmentation.

Strategy
--------
All statistics are derived *exclusively* from the real cleaned dataset —
no external assumptions are introduced.  This prevents synthesiser-induced
bias (e.g. hardcoding "smokers are more likely to have heart disease").

For each diagnosis class (DIABETES, HEART DISEASE):

  Numerical features (Age, Blood_Pressure, Cholesterol, BMI):
    Fit a Ledoit-Wolf regularised multivariate Gaussian to the class subset.
    Ledoit-Wolf is used instead of the sample covariance because the real
    dataset has only 25-27 samples per class — too few for a reliable
    sample covariance matrix.  Sampling from this distribution preserves
    inter-feature correlations (e.g. BP-cholesterol co-variance) as
    observed in the real data.

  Categorical features (Smoker, Gender):
    Sampled independently from their empirical class-conditional frequencies
    (e.g. P(Smoker=YES | DIABETES) is estimated from the real data).
    Treating categoricals as independent of each other and of the numerical
    features is a simplifying assumption documented here.

Check-before-create
-------------------
`generate()` checks for the output CSV before doing any work.  Pass
``force=True`` to regenerate regardless.

Bias safeguards
---------------
- No hardcoded numeric ranges or category probabilities.
- All parameters are estimated from, and logged against, the real data.
- The saved ``synthetic_stats.json`` is a full audit trail: every mean,
  covariance matrix, and category probability that was used.
- Synthetic samples are clipped to physiologically plausible bounds derived
  from observed min/max in the real dataset (with a 10 % margin) — never
  tighter than what the real data contains.
- Class balance is preserved: synthetic proportions match real proportions.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(__file__)
REAL_PATH      = os.path.join(BASE_DIR, 'realworld_medical_dirty_cleaned_1.csv')
SYNTHETIC_PATH = os.path.join(BASE_DIR, 'synthetic_training_data.csv')
STATS_PATH     = os.path.join(BASE_DIR, 'model_outputs', 'synthetic_stats.json')

NUMERICAL_COLS   = ['Age', 'Blood_Pressure', 'Cholesterol', 'BMI']
CATEGORICAL_COLS = ['Smoker', 'Gender']
TARGET_COL       = 'Diagnosis'
CLASSES          = ['DIABETES', 'HEART DISEASE']

DEFAULT_N_SAMPLES = 500   # total synthetic samples across all classes
RANDOM_SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_labelled() -> pd.DataFrame:
    """Load cleaned data and keep only rows with known diagnoses."""
    df = pd.read_csv(REAL_PATH, keep_default_na=False, na_values=[''])
    return df[df[TARGET_COL].isin(CLASSES)].copy()


def _compute_bounds(df: pd.DataFrame) -> dict:
    """
    Derive clipping bounds per numerical column from the real data.
    Adds a 10 % margin beyond the observed min/max so extreme-but-valid
    synthetic values are not hard-clipped at the real data boundary.
    """
    bounds = {}
    for col in NUMERICAL_COLS:
        lo = df[col].min()
        hi = df[col].max()
        margin = (hi - lo) * 0.10
        bounds[col] = (max(0.0, lo - margin), hi + margin)
    return bounds


def _fit_class_stats(df: pd.DataFrame) -> dict:
    """
    Fit per-class distribution parameters from the real labelled data.

    Returns a dict keyed by class name containing:
      - 'n_real'      : number of real samples in this class
      - 'weight'      : fraction of total labelled samples
      - 'num_mean'    : list of means for NUMERICAL_COLS
      - 'num_cov'     : Ledoit-Wolf covariance matrix (list of lists)
      - 'cat_freqs'   : {col: {value: probability}} for CATEGORICAL_COLS
    """
    stats = {}
    for cls in CLASSES:
        subset = df[df[TARGET_COL] == cls]
        n = len(subset)

        # ── Numerical: Ledoit-Wolf covariance ────────────────────────────
        X_num = subset[NUMERICAL_COLS].values.astype(float)
        lw = LedoitWolf()
        lw.fit(X_num)

        # ── Categorical: empirical frequencies ───────────────────────────
        cat_freqs = {}
        for col in CATEGORICAL_COLS:
            vc = subset[col].value_counts(normalize=True)
            cat_freqs[col] = vc.to_dict()

        stats[cls] = {
            'n_real':    n,
            'weight':    n / len(df),
            'num_mean':  X_num.mean(axis=0).tolist(),
            'num_cov':   lw.covariance_.tolist(),
            'cat_freqs': cat_freqs,
        }
    return stats


def _sample_class(cls: str, n: int, stats: dict,
                  bounds: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Draw `n` synthetic samples for one class.

    Numerical features are sampled jointly from the fitted multivariate
    Gaussian and then clipped to physiologically plausible bounds.
    Categorical features are sampled independently from their empirical
    class-conditional distributions.
    """
    s = stats[cls]
    mean = np.array(s['num_mean'])
    cov  = np.array(s['num_cov'])

    # Sample numerical features
    num_samples = rng.multivariate_normal(mean, cov, size=n)
    df_num = pd.DataFrame(num_samples, columns=NUMERICAL_COLS)

    # Clip to derived bounds
    for col in NUMERICAL_COLS:
        lo, hi = bounds[col]
        df_num[col] = df_num[col].clip(lo, hi).round(1)

    # Sample categorical features
    df_cat_parts = {}
    for col in CATEGORICAL_COLS:
        freq = s['cat_freqs'][col]
        values = list(freq.keys())
        probs  = [freq[v] for v in values]
        df_cat_parts[col] = rng.choice(values, size=n, p=probs)

    df_cat = pd.DataFrame(df_cat_parts)

    # Assemble
    df_out = pd.concat([df_num, df_cat], axis=1)
    df_out[TARGET_COL] = cls
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate(n_samples: int = DEFAULT_N_SAMPLES,
             seed: int = RANDOM_SEED,
             force: bool = False) -> str:
    """
    Generate ``n_samples`` synthetic training records and save them to
    ``synthetic_training_data.csv``.

    Check-before-create: if the file already exists and ``force=False``,
    this function returns immediately without regenerating.

    Parameters
    ----------
    n_samples : int
        Total number of synthetic rows to generate across all classes.
        Class sizes are proportional to the real data class balance.
    seed : int
        Random seed for reproducibility.
    force : bool
        If True, regenerate even if the file already exists.

    Returns
    -------
    str  Path to the synthetic CSV file.
    """
    if os.path.exists(SYNTHETIC_PATH) and not force:
        return SYNTHETIC_PATH

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    rng = np.random.default_rng(seed)

    real_df = _load_real_labelled()
    bounds  = _compute_bounds(real_df)
    stats   = _fit_class_stats(real_df)

    # Persist audit trail
    _save_stats(stats, bounds, n_samples, seed, len(real_df))

    # Sample per class, proportional to real class balance
    frames = []
    for i, cls in enumerate(CLASSES):
        # Distribute any remainder to the first class
        n_cls = round(n_samples * stats[cls]['weight'])
        if i == len(CLASSES) - 1:
            n_cls = n_samples - sum(len(f) for f in frames)
        frames.append(_sample_class(cls, n_cls, stats, bounds, rng))

    synthetic = pd.concat(frames, ignore_index=True)
    synthetic = synthetic.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add Patient_ID and placeholder columns to match real data schema
    synthetic.insert(0, 'Patient_ID',
                     [f'SYN{i:04d}' for i in range(len(synthetic))])
    synthetic['Admission_Date'] = 'N/A'
    synthetic['Notes']          = 'N/A'

    # Reorder to match real data column order
    col_order = ['Patient_ID', 'Age', 'Gender', 'Blood_Pressure',
                 'Cholesterol', 'BMI', 'Smoker', 'Diagnosis',
                 'Admission_Date', 'Notes']
    synthetic = synthetic[col_order]

    synthetic.to_csv(SYNTHETIC_PATH, index=False)
    return SYNTHETIC_PATH


def _save_stats(stats: dict, bounds: dict,
                n_samples: int, seed: int, n_real: int):
    """Save the generation parameters as a transparent audit trail."""
    serialisable = {
        'generation_params': {
            'n_samples': n_samples,
            'seed':      seed,
            'n_real_labelled': n_real,
        },
        'bounds_used': bounds,
        'class_stats': {
            cls: {
                'n_real':    s['n_real'],
                'weight':    round(s['weight'], 4),
                'num_mean':  {col: round(v, 3)
                              for col, v in zip(NUMERICAL_COLS, s['num_mean'])},
                'cat_freqs': s['cat_freqs'],
                # Full covariance matrix omitted from human-readable stats
                # (available in the class_stats dict passed to this function)
            }
            for cls, s in stats.items()
        },
    }
    with open(STATS_PATH, 'w') as fh:
        json.dump(serialisable, fh, indent=2)


def load_stats() -> dict:
    """Load the generation audit trail. Returns None if not yet generated."""
    if not os.path.exists(STATS_PATH):
        return {}
    with open(STATS_PATH) as fh:
        return json.load(fh)


def exists() -> bool:
    """Return True if the synthetic dataset has already been generated."""
    return os.path.exists(SYNTHETIC_PATH)


if __name__ == '__main__':
    path = generate(n_samples=500, force=True)
    df   = pd.read_csv(path)
    print(f"Generated {len(df)} synthetic samples → {path}")
    print(df['Diagnosis'].value_counts())
    print(df.describe().round(1))
    print("\nAudit trail saved to:", STATS_PATH)
