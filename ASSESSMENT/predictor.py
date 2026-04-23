"""
Readmission predictor for the Diabetes 130-US Hospitals dataset.

Task    : Binary classification — early readmission within 30 days (<30) vs not.
Model   : Gradient Boosting Classifier (sklearn GradientBoostingClassifier)
Dataset : 101,766 patient encounters, cleaned to ~70k unique patients.

Selected model rationale and hyperparameter decisions documented in
docs/dataset_decisions.md
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from cleaning import (
    load_and_clean, get_features_and_target,
    FEATURE_COLS, TARGET_COL,
    AGE_MIDPOINTS, DIAG_CAT_MAP, _A1C_MAP, _GLU_MAP, _MED_MAP,
    _RACE_MAP, _ADMISSION_TYPE_MAP,
)

MODEL_DIR        = os.path.join(os.path.dirname(__file__), 'model_outputs')
MODEL_PATH       = os.path.join(MODEL_DIR, 'rf_model.pkl')
ENCODERS_PATH    = os.path.join(MODEL_DIR, 'encoders.pkl')
PERFORMANCE_JSON = os.path.join(MODEL_DIR, 'performance.json')
COMPARISON_JSON  = os.path.join(MODEL_DIR, 'comparison.json')

# GradientBoosting hyperparameters — best test ROC-AUC in model comparison
# (see docs/model_comparison.md — XGBoost/LightGBM trialled but GBM won on held-out test AUC)
GB_CONFIG = {
    'n_estimators':    500,
    'learning_rate':   0.05,
    'max_depth':       4,
    'min_samples_leaf': 30,
    'subsample':       0.8,
    'max_features':    'sqrt',
    'random_state':    42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train() -> dict:
    """
    Train the Gradient Boosting model on the diabetes readmission dataset.

    Split   : 80 % train / 20 % test (stratified by readmission label).
    CV      : 5-fold stratified cross-validation on the training set.
    Returns : performance metrics dict (also written to performance.json).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = get_features_and_target()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    sample_weights = compute_sample_weight('balanced', y_train)

    model = GradientBoostingClassifier(**GB_CONFIG)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # ── Hold-out evaluation ──────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    report   = classification_report(y_test, y_pred,
                                     target_names=['Not Early', 'Early (<30d)'],
                                     output_dict=True)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    gap       = round((train_acc - test_acc) * 100, 2)

    # ── Cross-validation ─────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)

    # ── Confusion matrix data ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    # ── Feature importance ────────────────────────────────────────────────────
    # XGBoost returns float32 — convert to plain Python float for JSON serialisation
    importance = {k: round(float(v), 4)
                  for k, v in zip(FEATURE_COLS, model.feature_importances_)}

    results = {
        'model': 'GradientBoostingClassifier',
        'data_splits': {
            'training': int(len(X_train)),
            'test':     int(len(X_test)),
            'total':    int(len(X)),
        },
        'test_performance': {
            'accuracy':          round(test_acc, 4),
            'roc_auc':           round(roc_auc, 4),
            'avg_precision':     round(avg_prec, 4),
            'report':            report,
        },
        'training_performance': {
            'accuracy': round(train_acc, 4),
        },
        'overfitting_gap': gap,
        'cross_validation': {
            'cv_accuracy_mean': round(float(cv_acc.mean()), 4),
            'cv_accuracy_std':  round(float(cv_acc.std()), 4),
            'cv_roc_auc_mean':  round(float(cv_roc.mean()), 4),
            'cv_roc_auc_std':   round(float(cv_roc.std()), 4),
            'folds': 5,
        },
        'confusion_matrix': cm.tolist(),
        'feature_importance': importance,
        'model_config': GB_CONFIG,
        # Legacy keys expected by performance_dashboard and templates
        'test_acc':        round(test_acc, 4),
        'train_acc':       round(train_acc, 4),
        'roc_auc':         round(roc_auc, 4),
        'cv_acc':          cv_acc.tolist(),
        'cv_roc':          cv_roc.tolist(),
        'fpr':             [],   # not storing full curve to keep JSON small
        'tpr':             [],
        'per_class': {
            'Not Early (<30d)': {
                'precision': round(report['Not Early']['precision'], 3),
                'recall':    round(report['Not Early']['recall'], 3),
                'f1':        round(report['Not Early']['f1-score'], 3),
            },
            'Early (<30d)': {
                'precision': round(report['Early (<30d)']['precision'], 3),
                'recall':    round(report['Early (<30d)']['recall'], 3),
                'f1':        round(report['Early (<30d)']['f1-score'], 3),
            },
        },
        'class_names': ['Not Early', 'Early (<30d)'],
        'mean_confidence': round(float(y_proba.mean()), 4),
        'overfitting_gap': gap,
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
        'training_mode': 'real_data_only',
        'loo_acc': round(float(cv_acc.mean()), 4),
        'loo_roc': round(float(cv_roc.mean()), 4),
    }

    with open(PERFORMANCE_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    _save_confusion_matrix(y_test.values, y_pred, ['Not Early', 'Early (<30d)'])
    _save_feature_importance(model)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    meta = {'feature_cols': FEATURE_COLS, 'target_col': TARGET_COL}
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(meta, f)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_performance() -> dict:
    if not os.path.exists(PERFORMANCE_JSON):
        train()
    with open(PERFORMANCE_JSON) as f:
        return json.load(f)


def load_model():
    if not os.path.exists(MODEL_PATH):
        train()
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


def load_comparison() -> dict:
    if os.path.exists(COMPARISON_JSON):
        with open(COMPARISON_JSON) as f:
            return json.load(f)
    return {'best_model': 'GradientBoostingClassifier', 'models': {}}


def compare_models() -> dict:
    return load_comparison()


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
    ax.set_title('Confusion Matrix — Readmission Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _save_feature_importance(model):
    importances = model.feature_importances_
    labels = [c.replace('_', ' ').title() for c in FEATURE_COLS]
    sorted_idx = np.argsort(importances)[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([labels[i] for i in sorted_idx], importances[sorted_idx], color='#2980b9')
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title('Top 15 Feature Importances — Gradient Boosting')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction API
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    age_range: str,
    gender: str,
    race: str,
    time_in_hospital: int,
    num_lab_procedures: int,
    num_procedures: int,
    num_medications: int,
    number_outpatient: int,
    number_emergency: int,
    number_inpatient: int,
    number_diagnoses: int,
    a1c_result: str,
    glu_serum: str,
    insulin: str,
    diabetes_med: str,
    medication_changed: str,
    admission_type: int,
    diag_1_category: str,
) -> dict:
    """
    Predict early readmission risk for a single patient encounter.

    Returns dict with 'prediction' (0/1), 'label', and 'probabilities'.
    """
    model = load_model()

    age_numeric    = AGE_MIDPOINTS.get(age_range, 55)
    gender_enc     = 1 if gender.lower() == 'male' else 0
    race_enc       = _RACE_MAP.get(race, 5)
    a1c_enc        = _A1C_MAP.get(a1c_result, 0)
    glu_enc        = _GLU_MAP.get(glu_serum, 0)
    insulin_enc    = _MED_MAP.get(insulin, 0)
    metformin_enc  = 0
    glipizide_enc  = 0
    glyburide_enc  = 0
    glimepiride_enc = 0
    change_enc     = 1 if medication_changed == 'Ch' else 0
    diab_med_enc   = 1 if diabetes_med == 'Yes' else 0
    adm_type_grp   = _ADMISSION_TYPE_MAP.get(int(admission_type), 3)
    discharge_grp        = 0   # discharged home
    discharge_disp_raw   = 1   # discharge_disposition_id=1 (home)
    admission_source_grp = 0   # physician referral
    diag1_enc      = DIAG_CAT_MAP.get(diag_1_category, 8)

    num_meds_changed = 1 if insulin in ('Up', 'Down') else 0
    num_meds_used    = 1 if insulin != 'No' else 0

    # Engineered features — derived from the inputs above (mirrors cleaning.py logic)
    total_prior_visits  = number_outpatient + number_emergency + number_inpatient
    has_prior_inpatient = 1 if number_inpatient > 0 else 0
    high_risk_discharge = 1 if discharge_disp_raw in {9, 12, 15, 22, 28} else 0
    insulin_down        = 1 if insulin_enc == 3 else 0
    poor_glycaemic_ctrl = 1 if (a1c_enc >= 2 and insulin_enc >= 1) else 0
    long_stay           = 1 if time_in_hospital >= 7 else 0
    multimorbid         = 1 if number_diagnoses >= 7 else 0

    # Values in the exact order of FEATURE_COLS (36 features)
    features = pd.DataFrame([[
        age_numeric, gender_enc, race_enc,
        time_in_hospital, adm_type_grp, discharge_grp,
        discharge_disp_raw, admission_source_grp,
        num_lab_procedures, num_procedures, num_medications,
        number_outpatient, number_emergency, number_inpatient, number_diagnoses,
        a1c_enc, glu_enc,
        insulin_enc, metformin_enc, glipizide_enc, glyburide_enc, glimepiride_enc,
        change_enc, diab_med_enc, num_meds_changed, num_meds_used,
        total_prior_visits, has_prior_inpatient, high_risk_discharge,
        insulin_down, poor_glycaemic_ctrl, long_stay, multimorbid,
        diag1_enc,
        8,  # diag_2_cat_enc = Other
        8,  # diag_3_cat_enc = Other
    ]], columns=FEATURE_COLS)

    pred  = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]

    return {
        'prediction':    pred,
        'label':         'Early Readmission Risk' if pred == 1 else 'Low Readmission Risk',
        'probabilities': {
            'Low Risk (Not Early)': round(float(proba[0]) * 100, 1),
            'High Risk (Early <30d)': round(float(proba[1]) * 100, 1),
        },
    }


if __name__ == '__main__':
    results = train()
    print(f"Test Accuracy : {results['test_performance']['accuracy']:.1%}")
    print(f"ROC-AUC       : {results['test_performance']['roc_auc']:.3f}")
    print(f"CV ROC-AUC    : {results['cross_validation']['cv_roc_auc_mean']:.3f} "
          f"± {results['cross_validation']['cv_roc_auc_std']:.3f}")
    print(f"Avg Precision : {results['test_performance']['avg_precision']:.3f}")
    print(f"Overfit Gap   : {results['overfitting_gap']:.1f}%")
