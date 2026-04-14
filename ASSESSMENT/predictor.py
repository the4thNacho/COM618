"""
Predictive model for medical diagnosis.

Trains a Random Forest classifier on the cleaned dataset to predict whether
a patient has DIABETES or HEART DISEASE based on their clinical features.
Patients labelled UNKNOWN are excluded from training (no ground-truth label).

Improvements over the baseline:
  - GridSearchCV tunes max_depth / min_samples_leaf to reduce overfitting.
  - Leave-One-Out CV gives a less noisy accuracy estimate on the small dataset.
  - compare_models() benchmarks RF against LR, SVM, and Naive Bayes.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

CLEANED_PATH    = os.path.join(os.path.dirname(__file__), 'realworld_medical_dirty_cleaned_1.csv')
MODEL_DIR       = os.path.join(os.path.dirname(__file__), 'model_outputs')
MODEL_PATH      = os.path.join(MODEL_DIR, 'knn_model.pkl')
ENCODERS_PATH   = os.path.join(MODEL_DIR, 'encoders.pkl')
PERFORMANCE_JSON = os.path.join(MODEL_DIR, 'performance.json')
COMPARISON_JSON  = os.path.join(MODEL_DIR, 'comparison.json')

FEATURE_COLS = ['Age', 'Blood_Pressure', 'Cholesterol', 'BMI', 'Smoker_enc', 'Gender_enc']
TARGET_COL   = 'Diagnosis'

# GridSearchCV parameter grid for the Random Forest
RF_PARAM_GRID = {
    'max_depth':        [2, 3, 4, 5, None],
    'min_samples_leaf': [1, 2, 3, 5],
    'n_estimators':     [50, 100],
}

# GridSearchCV parameter grid for K-Nearest Neighbors
KNN_PARAM_GRID = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_prepare(path: str | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Load a labelled CSV, encode categoricals, and return (X, y, encoders).

    Parameters
    ----------
    path : str or None
        CSV to load.  Defaults to the real cleaned dataset.
        Pass a synthetic CSV path to prepare synthetic training data
        using encoders fitted on the real data distribution.
    """
    # Always fit encoders on the real data so label mappings are consistent
    real_df = pd.read_csv(CLEANED_PATH, keep_default_na=False, na_values=[''])
    real_df = real_df[real_df[TARGET_COL] != 'UNKNOWN'].copy()

    le_smoker = LabelEncoder().fit(real_df['Smoker'].str.upper())
    le_gender = LabelEncoder().fit(real_df['Gender'].str.upper())
    le_target = LabelEncoder().fit(real_df[TARGET_COL].str.upper())
    encoders  = {'smoker': le_smoker, 'gender': le_gender, 'target': le_target}

    # Now load whichever dataset was requested
    src = path if path is not None else CLEANED_PATH
    df  = pd.read_csv(src, keep_default_na=False, na_values=[''])
    df  = df[df[TARGET_COL].isin(le_target.classes_)].copy()

    df['Smoker_enc'] = le_smoker.transform(df['Smoker'].str.upper())
    df['Gender_enc'] = le_gender.transform(df['Gender'].str.upper())
    y = le_target.transform(df[TARGET_COL].str.upper())

    return df[FEATURE_COLS], pd.Series(y, name=TARGET_COL), encoders


def _scaled_pipeline(model) -> Pipeline:
    """Wrap a model in a StandardScaler pipeline (required for LR and SVM)."""
    return Pipeline([('scaler', StandardScaler()), ('model', model)])


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison (runs all candidate models, saves comparison.json)
# ─────────────────────────────────────────────────────────────────────────────

def compare_models() -> dict:
    """
    Benchmark five supervised learning models using 5-fold CV.

    Candidates
    ----------
    RF (default)     — baseline, fully grown trees (demonstrates overfitting)
    RF (constrained) — max_depth=3, min_samples_leaf=4
    RF (tuned)       — best params found by GridSearchCV on CV ROC-AUC
    Logistic Reg.    — linear, L2 regularisation; needs feature scaling
    SVM (RBF)        — kernel SVM; needs feature scaling
    Naive Bayes      — Gaussian NB; very low variance on small datasets

    Returns
    -------
    dict  keyed by model name, each value contains cv_acc_mean, cv_acc_std,
          cv_roc_mean, cv_roc_std (all %).  Also includes 'best_model' and
          'best_params' keys.  Saved to model_outputs/comparison.json.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Use synthetic training data for comparison if available
    from synthesise import generate as _gen_synthetic
    train_path = _gen_synthetic()  # handles check-before-create internally
    X, y, _ = _load_and_prepare(train_path)

    # GridSearchCV to find optimal KNN hyperparameters
    cv_folds = min(5, len(y) // 2)   # guard against tiny datasets
    
    # KNN pipeline for grid search
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    knn_param_grid = {f'knn__{k}': v for k, v in KNN_PARAM_GRID.items()}
    
    grid = GridSearchCV(
        knn_pipeline,
        knn_param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
    )
    grid.fit(X, y)
    best_knn_params = grid.best_params_
    
    # Also find best RF params for comparison
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_PARAM_GRID,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
    )
    rf_grid.fit(X, y)
    best_rf_params = rf_grid.best_params_

    candidates = {
        'KNN (tuned)':      Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**{k.replace('knn__', ''): v for k, v in best_knn_params.items()}))
        ]),
        'KNN (k=3)':        _scaled_pipeline(KNeighborsClassifier(n_neighbors=3, weights='distance')),
        'KNN (k=5)':        _scaled_pipeline(KNeighborsClassifier(n_neighbors=5, weights='distance')),
        'RF (default)':     RandomForestClassifier(n_estimators=100, random_state=42),
        'RF (constrained)': RandomForestClassifier(n_estimators=100, max_depth=3,
                                                    min_samples_leaf=4, random_state=42),
        'RF (tuned)':       RandomForestClassifier(**best_rf_params, random_state=42),
        'Logistic Reg.':    _scaled_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
        'SVM (RBF)':        _scaled_pipeline(SVC(kernel='rbf', probability=True, random_state=42)),
        'Naive Bayes':      GaussianNB(),
    }

    results = {}
    for name, model in candidates.items():
        cv_acc = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_roc = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
        results[name] = {
            'cv_acc_mean': round(float(cv_acc.mean()) * 100, 1),
            'cv_acc_std':  round(float(cv_acc.std())  * 100, 1),
            'cv_roc_mean': round(float(cv_roc.mean()) * 100, 1),
            'cv_roc_std':  round(float(cv_roc.std())  * 100, 1),
        }

    # Identify best model by CV ROC-AUC
    best_name = max(results, key=lambda n: results[n]['cv_roc_mean'])

    output = {
        'models':           results,
        'best_model':       best_name,
        'best_knn_params':  best_knn_params,
        'best_rf_params':   best_rf_params,
    }
    with open(COMPARISON_JSON, 'w') as fh:
        json.dump(output, fh, indent=2)

    return output


def load_comparison() -> dict:
    """Load saved model comparison. Runs compare_models() if not found."""
    if not os.path.exists(COMPARISON_JSON):
        return compare_models()
    with open(COMPARISON_JSON) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Training (uses GridSearchCV-tuned RF, adds LOO CV)
# ─────────────────────────────────────────────────────────────────────────────

def train() -> dict:
    """
    Train the K-Nearest Neighbors model (hyperparameters tuned via GridSearchCV)
    and save it to disk.

    Training data
    -------------
    If ``synthetic_training_data.csv`` exists the model is trained on that
    larger dataset.  The 52 real labelled samples are always used as the
    held-out test set, giving an honest evaluation of whether the model
    generalises to real patients.

    If no synthetic data is found the model falls back to a 75/25 split of
    the real labelled data (original behaviour).

    Metrics recorded
    ----------------
    - Train / test accuracy and overfitting gap
    - ROC-AUC (hold-out split + 5-fold CV + Leave-One-Out CV)
    - Per-class precision, recall, F1
    - Prediction confidence distribution
    - Best hyperparameters found by GridSearchCV
    - training_mode: 'synthetic' or 'real_only'
    """
    from synthesise import generate as _gen_synthetic, exists as _syn_exists

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Always load real data — used for test set and for fitting encoders
    X_real, y_real, encoders = _load_and_prepare()
    class_names = list(encoders['target'].classes_)

    use_synthetic = _syn_exists()

    if use_synthetic:
        # Train on synthetic; evaluate on all 52 real labelled samples
        syn_path = _gen_synthetic()           # no-op if already exists
        X_train, y_train, _ = _load_and_prepare(syn_path)
        X_test,  y_test     = X_real, y_real
        training_mode = 'synthetic'
    else:
        # Fall back: 75/25 split of real data
        X_train, X_test, y_train, y_test = train_test_split(
            X_real, y_real, test_size=0.25, random_state=42, stratify=y_real
        )
        training_mode = 'real_only'

    # ── GridSearchCV on training data ─────────────────────────────────────
    cv_folds = min(5, len(y_train) // 2)
    
    # KNN needs feature scaling, so we use a pipeline
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    
    # Adjust param grid for pipeline (need 'knn__' prefix)
    knn_param_grid = {
        f'knn__{k}': v for k, v in KNN_PARAM_GRID.items()
    }
    
    grid = GridSearchCV(
        knn_pipeline,
        knn_param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    # Create the best KNN model with scaling pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(**{k.replace('knn__', ''): v for k, v in best_params.items()}))
    ])
    model.fit(X_train, y_train)

    # ── Predictions ───────────────────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)
    y_train_pred = model.predict(X_train)

    # ── Core accuracy ─────────────────────────────────────────────────────
    test_acc  = float(accuracy_score(y_test, y_pred))
    train_acc = float(accuracy_score(y_train, y_train_pred))

    # ── ROC-AUC ───────────────────────────────────────────────────────────
    roc_auc_split = float(roc_auc_score(y_test, y_proba[:, 1]))
    fpr, tpr, _   = roc_curve(y_test, y_proba[:, 1])

    # ── 5-fold CV on training data ────────────────────────────────────────
    cv_folds_eval = min(5, len(y_train) // 2)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv_folds_eval, scoring='accuracy')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv_folds_eval, scoring='roc_auc')

    # ── Leave-One-Out CV on real labelled data ────────────────────────────
    # Always runs on the 52 real samples for a consistent comparison,
    # regardless of whether the model was trained on synthetic data.
    loo = LeaveOneOut()
    loo_preds  = []
    loo_labels = []
    loo_probas = []
    
    # Extract KNN params without the 'knn__' prefix for LOO
    knn_params = {k.replace('knn__', ''): v for k, v in best_params.items()}
    
    for train_idx, test_idx in loo.split(X_real):
        # Create KNN pipeline for each LOO fold
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**knn_params))
        ])
        m.fit(X_real.iloc[train_idx], y_real.iloc[train_idx])
        loo_preds.append(int(m.predict(X_real.iloc[test_idx])[0]))
        loo_labels.append(int(y_real.iloc[test_idx].iloc[0]))
        loo_probas.append(float(m.predict_proba(X_real.iloc[test_idx])[0][1]))

    loo_acc = float(accuracy_score(loo_labels, loo_preds))
    loo_roc = float(roc_auc_score(loo_labels, loo_probas))

    # ── Per-class metrics ─────────────────────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )
    per_class = {
        cls: {
            'precision': round(float(p) * 100, 1),
            'recall':    round(float(r) * 100, 1),
            'f1':        round(float(f) * 100, 1),
        }
        for cls, p, r, f in zip(class_names, precision, recall, f1)
    }

    # ── Confidence ────────────────────────────────────────────────────────
    confidence_scores = [round(float(max(row)) * 100, 1) for row in y_proba]
    mean_confidence   = round(float(np.mean([max(row) for row in y_proba])) * 100, 1)

    # ── Persist ───────────────────────────────────────────────────────────
    perf_data = {
        'y_test':           y_test.tolist(),
        'y_pred':           y_pred.tolist(),
        'y_proba':          y_proba.tolist(),
        'fpr':              fpr.tolist(),
        'tpr':              tpr.tolist(),
        'class_names':      class_names,
        'train_acc':        train_acc,
        'test_acc':         test_acc,
        'roc_auc':          roc_auc_split,
        'cv_acc':           cv_acc.tolist(),
        'cv_roc':           cv_roc.tolist(),
        'loo_acc':          loo_acc,
        'loo_roc':          loo_roc,
        'confidence_scores': confidence_scores,
        'mean_confidence':  mean_confidence,
        'per_class':        per_class,
        'overfitting_gap':  round((train_acc - test_acc) * 100, 1),
        'best_params':      best_params,
        'training_mode':    training_mode,
        'n_train':          len(y_train),
        'n_test':           len(y_test),
    }
    with open(PERFORMANCE_JSON, 'w') as fh:
        json.dump(perf_data, fh, indent=2)

    with open(MODEL_PATH, 'wb') as fh:
        pickle.dump(model, fh)
    with open(ENCODERS_PATH, 'wb') as fh:
        pickle.dump(encoders, fh)

    _save_confusion_matrix(y_test, y_pred, class_names)
    # Note: KNN doesn't have feature_importances_ like Random Forest
    # Could implement permutation importance in future if needed
    # _save_feature_importance(model)

    return {
        'accuracy':         round(test_acc  * 100, 1),
        'train_accuracy':   round(train_acc * 100, 1),
        'overfitting_gap':  round((train_acc - test_acc) * 100, 1),
        'roc_auc':          round(roc_auc_split * 100, 1),
        'cv_roc_auc_mean':  round(float(cv_roc.mean()) * 100, 1),
        'cv_roc_auc_std':   round(float(cv_roc.std())  * 100, 1),
        'cv_mean':          round(float(cv_acc.mean()) * 100, 1),
        'cv_std':           round(float(cv_acc.std())  * 100, 1),
        'cv_scores':        [round(s * 100, 1) for s in cv_acc.tolist()],
        'cv_roc_scores':    [round(s * 100, 1) for s in cv_roc.tolist()],
        'loo_acc':          round(loo_acc * 100, 1),
        'loo_roc':          round(loo_roc * 100, 1),
        'mean_confidence':  mean_confidence,
        'confidence_scores': confidence_scores,
        'per_class':        per_class,
        'best_params':      best_params,
        'training_mode':    training_mode,
        'n_train':          len(y_train),
        'n_test':           len(y_test),
        'report':           classification_report(y_test, y_pred, target_names=class_names),
        'classes':          class_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_performance() -> dict:
    """Load persisted performance data from JSON. Trains first if not found."""
    if not os.path.exists(PERFORMANCE_JSON):
        train()
    with open(PERFORMANCE_JSON) as fh:
        return json.load(fh)


def load_model() -> tuple:
    """Load the saved model and encoders. Trains first if not found."""
    if not os.path.exists(MODEL_PATH):
        train()
    with open(MODEL_PATH, 'rb') as fh:
        model = pickle.load(fh)
    with open(ENCODERS_PATH, 'rb') as fh:
        encoders = pickle.load(fh)
    return model, encoders


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _save_feature_importance(model):
    importances = model.feature_importances_
    labels      = ['Age', 'Blood Pressure', 'Cholesterol', 'BMI', 'Smoker', 'Gender']
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([labels[i] for i in sorted_idx], importances[sorted_idx], color='#3498db')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importances')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(age: float, blood_pressure: float, cholesterol: float,
            bmi: float, smoker: str, gender: str) -> dict:
    """
    Make a single prediction using the trained model.

    Returns dict with 'diagnosis', 'probabilities', and 'classes'.
    """
    model, encoders = load_model()

    smoker_enc = encoders['smoker'].transform([smoker.upper()])[0]
    gender_enc = encoders['gender'].transform([gender.upper()])[0]

    features = pd.DataFrame(
        [[age, blood_pressure, cholesterol, bmi, smoker_enc, gender_enc]],
        columns=FEATURE_COLS,
    )
    pred_idx = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]

    diagnosis = encoders['target'].inverse_transform([pred_idx])[0]
    classes   = list(encoders['target'].classes_)

    return {
        'diagnosis':     diagnosis,
        'probabilities': {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)},
        'classes':       classes,
    }


if __name__ == '__main__':
    print("Running model comparison...")
    comp = compare_models()
    print(f"Best model: {comp['best_model']}")
    print(f"Best RF params: {comp['best_params']}")
    for name, r in comp['models'].items():
        print(f"  {name:<22} CV acc={r['cv_acc_mean']}%  CV AUC={r['cv_roc_mean']}%")

    print("\nTraining final model...")
    metrics = train()
    print(f"Test accuracy:    {metrics['accuracy']}%")
    print(f"Train accuracy:   {metrics['train_accuracy']}%")
    print(f"Overfitting gap:  {metrics['overfitting_gap']}%")
    print(f"ROC-AUC (split):  {metrics['roc_auc']}%")
    print(f"CV acc:           {metrics['cv_mean']}% ± {metrics['cv_std']}%")
    print(f"CV ROC-AUC:       {metrics['cv_roc_auc_mean']}% ± {metrics['cv_roc_auc_std']}%")
    print(f"LOO accuracy:     {metrics['loo_acc']}%")
    print(f"LOO ROC-AUC:      {metrics['loo_roc']}%")
    print(f"Best params:      {metrics['best_params']}")
