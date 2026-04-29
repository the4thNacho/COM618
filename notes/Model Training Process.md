# Model Training Process

This document explains how the three models are trained, evaluated, and selected, including the reasoning behind each design decision. All code is from `predictor.py`.

---

## Overview

The pipeline trains three models on the same data split, compares them by ROC-AUC, and automatically selects the best as the active prediction model. No manual switching is needed — re-running training always updates the saved model.

```python
MODELS = {
    'Logistic Regression': LogisticRegression(...),
    'Random Forest':       RandomForestClassifier(...),
    'Gradient Boosting':   GradientBoostingClassifier(...),
}

for name, model in MODELS.items():
    model.fit(X_train, y_train)
    # evaluate, store results
    if roc_auc > best_auc:
        best_model = model
```

---

## Step 1 — Data Split

```python
from sklearn.model_selection import train_test_split

X, y = get_features_and_target()   # 70,439 patients, 36 features

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # keeps ~9% positive rate in both splits
)
# X_train: ~56,351   X_test: ~14,088
```

**Why 80/20?** With ~70k rows there is enough data for a 20% hold-out to give stable estimates without wasting training examples. `stratify=y` is critical — without it, random splits could produce a test set with significantly more or fewer positive cases, making evaluation unreliable.

---

## Step 2 — Handling Class Imbalance

Only ~9% of patients are readmitted within 30 days. Without correction, all three models would default to predicting "not readmitted" for nearly everything, achieving ~91% accuracy while being useless.

**Two approaches are used:**

### Balanced class weights (Logistic Regression, Random Forest)
```python
LogisticRegression(class_weight='balanced', ...)
RandomForestClassifier(class_weight='balanced', ...)
```
`class_weight='balanced'` automatically sets the weight of each class to be inversely proportional to its frequency:
```
weight_positive  = n_total / (2 * n_positive)   # ≈ 5.5x
weight_negative  = n_total / (2 * n_negative)   # ≈ 0.55x
```
This makes misclassifying a positive (rare) case ~10× more costly than misclassifying a negative.

### Sample weights (Gradient Boosting)
GradientBoostingClassifier does not accept `class_weight`, so balanced sample weights are computed and passed at fit time:
```python
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```
This achieves the same effect — each positive training example is upweighted by ~5.5×.

---

## Step 3 — Model Configurations

### Logistic Regression (linear baseline)

```python
LogisticRegression(
    C=0.1,          # inverse regularisation strength — lower = stronger L2 penalty
    max_iter=1000,  # needed to reach convergence on 36 features
    class_weight='balanced',
    solver='lbfgs', # efficient for medium-sized datasets
    random_state=42,
)
```

**Why C=0.1?** Stronger regularisation (small C) prevents overfitting on correlated features. Experiments with C=0.01, 0.1, and 1.0 produced identical ROC-AUC (0.637) — LR is at its linear ceiling regardless of regularisation on this dataset.

**Why no StandardScaler?** Tested — no improvement in ROC-AUC (0.637 with or without scaling). `lbfgs` is scale-invariant in practice and L2 regularisation already handles scale implicitly.

### Random Forest

```python
RandomForestClassifier(
    n_estimators=300,      # 300 trees — diminishing returns beyond this
    max_depth=8,           # prevents overfitting; deeper trees memorise
    min_samples_leaf=20,   # each leaf must have ≥20 samples
    max_features='sqrt',   # √36 ≈ 6 features per split — standard for RF
    class_weight='balanced',
    n_jobs=-1,             # parallelise across all CPU cores
    random_state=42,
)
```

**Why max_depth=8?** Tested depth=12 and depth=None (unlimited). Deeper trees pushed accuracy to 78–82% but ROC-AUC stayed at 0.657 — the extra accuracy was overfitting the majority class, not genuinely learning. `max_depth=8` gives consistent generalisation.

### Gradient Boosting (selected best)

```python
GradientBoostingClassifier(
    n_estimators=500,      # more trees than RF because each is shallow and weak
    learning_rate=0.03,    # small steps — each tree corrects only a little
    max_depth=4,           # deliberately shallow; boosting uses many weak learners
    min_samples_leaf=30,   # conservative leaf size to resist overfitting
    subsample=0.8,         # stochastic gradient boosting — 80% of rows per tree
    max_features='sqrt',   # √36 features per split
    random_state=42,
)
```

**Why n=500 with lr=0.03 instead of n=300 with lr=0.05?** Lower learning rate + more trees is a well-established improvement in boosting (Friedman, 2001). Each tree makes a smaller correction, reducing the risk of overshooting. Experiments confirmed: n=300/lr=0.05 gave ROC-AUC 0.6609; n=500/lr=0.03 gave 0.6639 — a modest but real improvement.

**Why subsample=0.8?** Stochastic subsampling (using 80% of training rows per tree) adds randomness that reduces correlation between trees and acts as a form of regularisation. It also speeds up training.

---

## Step 4 — Evaluation Metrics

```python
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc      = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)        # PRIMARY metric
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_minor = f1_score(y_test, y_pred, pos_label=1, average='binary')
```

**Primary metric: ROC-AUC.** Measures the model's ability to rank a positive case above a negative case at any threshold. Immune to class imbalance. A score of 0.664 means the model correctly ranks a readmitted patient above a non-readmitted patient 66.4% of the time.

**F1 (minority class).** Harmonic mean of precision and recall for the positive class only. Penalises both false positives (flagging healthy patients) and false negatives (missing readmissions). Used as a secondary metric to check practical usefulness.

**Why not accuracy alone?** A model predicting "never readmitted" achieves 91% accuracy on this dataset. This is useless — it catches zero early readmissions.

---

## Step 5 — Threshold Optimisation

The default prediction threshold of 0.5 is wrong for imbalanced data. With only 9% positives, most patients have predicted probabilities well below 0.5, so the model almost never fires.

```python
import numpy as np

# Search for threshold that maximises F1 on the minority class
thresholds = np.linspace(0.05, 0.95, 300)
f1s = [
    f1_score(y_test, (y_proba >= t).astype(int), pos_label=1, zero_division=0)
    for t in thresholds
]
opt_threshold = float(thresholds[np.argmax(f1s)])
y_pred_opt    = (y_proba >= opt_threshold).astype(int)
```

**Results — F1 minority at default (0.5) vs optimal threshold:**

| Model | Default (0.5) | Optimal | Threshold |
|-------|--------------|---------|-----------|
| Logistic Regression | 0.213 | 0.222 | 0.541 |
| Random Forest | 0.227 | 0.234 | 0.538 |
| **Gradient Boosting** | **0.232** | **0.239** | **0.580** |

The optimal threshold is stored in `performance.json` and used for all predictions:

```python
# In the prediction API
perf      = load_performance()
threshold = perf.get('optimal_threshold', 0.5)   # 0.58 for GBM
pred      = int(proba[1] >= threshold)
```

**Caveat:** Finding the optimal threshold on the test set is technically a form of data leakage — it uses the test labels to choose the threshold. In a production system, threshold selection should be done on a separate validation set. For this assessment the effect is small (0.5–1% F1 difference) and is acknowledged.

---

## Step 6 — Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc = cross_val_score(model, X_train, y_train,
                          cv=cv, scoring='roc_auc', n_jobs=1)

print(f"CV ROC-AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
```

5-fold CV is run on the training set only (never touching the test set). It confirms that the held-out test performance is not a lucky split.

**Final CV results:**

| Model | Test ROC-AUC | CV ROC-AUC |
|-------|-------------|-----------|
| Logistic Regression | 0.637 | 0.627 ± 0.008 |
| Random Forest | 0.657 | 0.644 ± 0.003 |
| Gradient Boosting | **0.664** | **0.647 ± 0.006** |

The small gap between test and CV scores confirms good generalisation — the model is not overfit to the specific test split.

---

## Step 7 — Model Selection and Saving

```python
# Select best by ROC-AUC
if roc_auc > best_auc:
    best_auc   = roc_auc
    best_model = model

# Save best model to disk
import pickle
with open('model_outputs/rf_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save full comparison for the UI
with open('model_outputs/comparison.json', 'w') as f:
    json.dump({'best_model': best_name, 'models': comparison}, f)
```

The saved model is loaded by the prediction API on every request. If the model file does not exist, `train_all_models()` is called automatically.

---

## Feature Engineering Experiments

Two additional feature engineering approaches were tested and compared. Results include overfitting gap (train ROC-AUC − test ROC-AUC) to detect introduced overfitting.

### Experiment 1 — Finer ICD-9 Diagnosis Groupings

**Hypothesis:** Expanding from 9 diagnosis categories to 25 (splitting Circulatory into Hypertension/Ischaemic Heart/Heart Failure/Cerebrovascular, Respiratory into Pneumonia/COPD, Neoplasms into Malignant/Benign, etc.) would add discriminative signal.

**Implementation:** 25-category mapping applied to all three diagnosis columns; ordinal-encoded 0–24.

**Result:**

| Config | Test ROC-AUC | CV ROC-AUC | Overfit Gap | F1 Minority |
|--------|-------------|-----------|------------|------------|
| GBM 9 categories (baseline) | **0.6639** | 0.6466 | +0.0446 | 0.238 |
| GBM 25 categories (fine) | 0.6624 | 0.6471 | +0.0507 | 0.237 |

**Conclusion: not implemented.** Fine groupings hurt test ROC-AUC by 0.0015 and increased the overfitting gap by 0.006. The likely reason: ordinal encoding of 25 categories still treats diagnosis as a linear scale (0 < 1 < 2 … < 24), which is semantically meaningless — Hypertension (2) is not "between" Diabetes T1 (1) and Ischaemic Heart (3). The model was fitting noise. One-hot encoding would be the correct representation, but would add 25 sparse binary columns — a future option.

---

### Experiment 2 — Polynomial Interaction Features for Logistic Regression

**Hypothesis:** LR's linear decision boundary can't capture interactions like "old patient AND high prior admissions". Adding pairwise interaction terms (`age × number_inpatient`, etc.) breaks the linear ceiling without switching model.

**Approach:** Rather than expanding all 36 features (630 interaction terms — slow, prone to overfitting), interactions were computed only on the **top 10 features by GBM importance**. This produces just 45 interaction terms and solves in seconds with lbfgs.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

LR_POLY_FEATURES = [
    'discharge_disposition_id', 'discharge_grp', 'high_risk_discharge',
    'number_inpatient', 'num_lab_procedures', 'age_numeric',
    'num_medications', 'has_prior_inpatient', 'diag_1_cat_enc',
    'time_in_hospital',
]

Pipeline([
    ('sc',   StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('lr',   LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced',
                                solver='lbfgs', random_state=42)),
])
```

`StandardScaler` is applied first because `PolynomialFeatures` multiplies raw feature values — without scaling, a feature with range 0–80 dominates a feature with range 0–1 in the product terms.

**Result:**

| Config | Test ROC-AUC | CV ROC-AUC | Overfit Gap | F1 Minority |
|--------|-------------|-----------|------------|------------|
| LR baseline (36 features, no poly) | 0.6373 | 0.6272 | −0.004 | 0.222 |
| LR + poly top-10, C=0.01 | 0.6443 | 0.6269 | −0.012 | 0.223 |
| LR + poly top-10, C=0.05 | **0.6448** | **0.6280** | −0.012 | 0.227 |
| LR + poly top-10, C=0.1 | 0.6446 | 0.6281 | −0.012 | 0.227 |
| LR + poly top-10, C=0.5 | 0.6444 | 0.6281 | −0.011 | 0.229 |

**Conclusion: implemented (C=0.05).** ROC-AUC improves from 0.6373 to 0.6448 (+0.0075). No overfitting introduced — the gap remains negative (slight underfitting, expected for LR). CV ROC-AUC also improves slightly (0.6272 → 0.6280), confirming the gain generalises.

LR still does not beat GBM (0.664) so GBM remains the best model, but the improvement makes LR a more meaningful baseline in the comparison.

---

## Final Model Comparison (After All Improvements)

| Model | Test ROC-AUC | CV ROC-AUC | Overfit Gap | F1 Minority | Threshold |
|-------|-------------|-----------|------------|------------|-----------|
| LR + poly interactions | 0.6448 | 0.628 ± 0.005 | −0.04% | 0.228 | 0.535 |
| Random Forest | 0.6574 | 0.644 ± 0.003 | +1.26% | 0.234 | 0.538 |
| **Gradient Boosting** | **0.6639** | **0.647 ± 0.006** | **+1.43%** | **0.239** | **0.580** |

GBM wins on all three key metrics. RF has the most stable CV variance (±0.003). LR has the lowest overfitting gap (effectively zero).

---

## What Was Tried and Didn't Help

| Experiment | Result | Reason |
|---|---|---|
| StandardScaler before LR (plain) | No change (0.637) | lbfgs is scale-invariant; L2 reg handles scale |
| LR C=0.01, 0.1, 1.0 (no poly) | Identical ROC-AUC (0.637) | LR at linear ceiling regardless of regularisation |
| Fine ICD-9 (25 categories, ordinal) | −0.0015 ROC-AUC, +0.006 gap | Ordinal encoding wrong for nominal categories; model fits noise |
| Poly LR on all 36 features | Slow (>10 min), no improvement | 630 interaction terms overwhelm the signal; stronger reg needed |
| RF max_depth=12 | Accuracy 78%, ROC-AUC unchanged | Extra accuracy = overfitting majority class |
| RF max_depth=None | Accuracy 83%, ROC-AUC 0.658 | Severe overfitting; CV gap widens |
| GBM n=300, depth=5 | ROC-AUC 0.660 | Slightly worse than depth=4 — too complex per tree |
| GBM n=400, lr=0.05 | ROC-AUC 0.661 | Less effective than slower learning rate |

**Key insight:** The performance ceiling is driven by the data, not the model. The available features carry limited signal for a partly behavioural outcome. Feature engineering that adds noise (fine ICD ordinal) or interaction terms beyond the top predictors does not help.

---

## Possible Further Improvements

These are not implemented but would be worth exploring with more time:

1. **Feature engineering from diagnosis codes.** Currently 716 ICD-9 codes are bucketed into 9 categories. Finer-grained groupings (e.g., 20–30 categories) or using all three diagnosis columns together as interaction features could add signal.

2. **Polynomial/interaction features for LR.** LR's ceiling is a linear boundary. Adding `PolynomialFeatures(degree=2, interaction_only=True)` to the LR pipeline would let it learn interactions (e.g., "old + many prior admissions") without switching to a tree model.

3. **Calibrated threshold on a validation fold.** Currently the optimal threshold is found on the test set (minor leakage). A proper implementation would hold out a calibration fold from the training set for threshold selection only.

https://jmai.amegroups.org/article/view/9179/html
