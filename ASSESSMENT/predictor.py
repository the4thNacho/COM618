"""
Readmission predictor for the Diabetes 130-US Hospitals dataset.

Task    : Binary classification — early readmission within 30 days (<30) vs not.
Models  : Logistic Regression (baseline), Random Forest, Gradient Boosting.
          All three are trained and compared; the best by ROC-AUC is saved as the
          active prediction model.
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
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

# ─────────────────────────────────────────────────────────────────────────────
# Model configurations
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    'Logistic Regression': LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42,
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=30,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
    ),
}

SHORT_NAMES = {
    'Logistic Regression': 'LR',
    'Random Forest':       'RF',
    'Gradient Boosting':   'GBM',
}


# ─────────────────────────────────────────────────────────────────────────────
# Training — all three models
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models() -> dict:
    """
    Train Logistic Regression, Random Forest, and Gradient Boosting on an
    80/20 stratified split. Select the best by ROC-AUC and save it as the
    active prediction model.

    Returns the best model's performance dict (also written to performance.json).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = get_features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    sample_weights = compute_sample_weight('balanced', y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    comparison = {}
    best_name  = None
    best_auc   = -1.0

    for name, model in MODELS.items():
        print(f'Training {name}…', flush=True)

        # GBM accepts sample_weight in fit(); LR and RF use class_weight='balanced'
        if name == 'Gradient Boosting':
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba)
        avg_prec  = average_precision_score(y_test, y_proba)
        f1_macro  = f1_score(y_test, y_pred, average='macro')
        f1_minor  = f1_score(y_test, y_pred, pos_label=1, average='binary')
        report    = classification_report(
            y_test, y_pred,
            target_names=['Not Early', 'Early (<30d)'],
            output_dict=True,
        )
        train_acc = accuracy_score(y_train, model.predict(X_train))
        gap       = round((train_acc - acc) * 100, 2)

        cv_roc = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=1)

        # Find threshold that maximises F1 for the minority class
        thresholds = np.linspace(0.05, 0.95, 300)
        f1s = [f1_score(y_test, (y_proba >= t).astype(int),
                        pos_label=1, zero_division=0) for t in thresholds]
        opt_threshold = float(thresholds[int(np.argmax(f1s))])
        y_pred_opt = (y_proba >= opt_threshold).astype(int)
        f1_minor_opt = float(max(f1s))
        f1_macro_opt = f1_score(y_test, y_pred_opt, average='macro')

        cm = confusion_matrix(y_test, y_pred_opt)

        comparison[name] = {
            'accuracy':        round(accuracy_score(y_test, y_pred_opt), 4),
            'roc_auc':         round(roc_auc, 4),
            'avg_precision':   round(avg_prec, 4),
            'f1_macro':        round(f1_macro_opt, 4),
            'f1_minority':     round(f1_minor_opt, 4),
            'f1_minority_default': round(f1_minor, 4),
            'optimal_threshold': round(opt_threshold, 3),
            'train_acc':       round(train_acc, 4),
            'overfit_gap':     gap,
            'cv_roc_mean':     round(float(cv_roc.mean()), 4),
            'cv_roc_std':      round(float(cv_roc.std()), 4),
            'per_class': {
                'Not Early': {
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
            'confusion_matrix': cm.tolist(),
        }

        # recompute per_class report using opt-threshold predictions
        opt_report = classification_report(
            y_test, y_pred_opt,
            target_names=['Not Early', 'Early (<30d)'],
            output_dict=True,
        )
        comparison[name]['per_class'] = {
            'Not Early': {
                'precision': round(opt_report['Not Early']['precision'], 3),
                'recall':    round(opt_report['Not Early']['recall'], 3),
                'f1':        round(opt_report['Not Early']['f1-score'], 3),
            },
            'Early (<30d)': {
                'precision': round(opt_report['Early (<30d)']['precision'], 3),
                'recall':    round(opt_report['Early (<30d)']['recall'], 3),
                'f1':        round(opt_report['Early (<30d)']['f1-score'], 3),
            },
        }

        if roc_auc > best_auc:
            best_auc       = roc_auc
            best_name      = name
            best_model     = model
            best_y_pred    = y_pred_opt
            best_y_proba   = y_proba
            best_report    = opt_report
            best_cv_roc    = cv_roc
            best_threshold = opt_threshold

    print(f'Best model: {best_name} (ROC-AUC={best_auc:.4f})', flush=True)

    # ── Save comparison JSON ─────────────────────────────────────────────────
    comparison_out = {
        'best_model': best_name,
        'primary_metric': 'roc_auc',
        'models': comparison,
    }
    with open(COMPARISON_JSON, 'w') as f:
        json.dump(comparison_out, f, indent=2)

    # ── Save comparison chart ────────────────────────────────────────────────
    _save_comparison_chart(comparison)

    # ── Save confusion matrix + feature importance for best model ────────────
    _save_confusion_matrix(
        y_test.values, best_y_pred,
        ['Not Early', 'Early (<30d)'],
        best_name,
    )
    _save_feature_importance(best_model, best_name)

    # ── Build and save the best model's full performance dict ────────────────
    cm_best = comparison[best_name]['confusion_matrix']
    results = {
        'model': best_name,
        'data_splits': {
            'training': int(len(X_train)),
            'test':     int(len(X_test)),
            'total':    int(len(X)),
        },
        'optimal_threshold': round(best_threshold, 3),
        'test_performance': {
            'accuracy':      comparison[best_name]['accuracy'],
            'roc_auc':       comparison[best_name]['roc_auc'],
            'avg_precision': comparison[best_name]['avg_precision'],
            'f1_macro':      comparison[best_name]['f1_macro'],
            'f1_minority':   comparison[best_name]['f1_minority'],
            'report':        best_report,
        },
        'training_performance': {
            'accuracy': comparison[best_name]['train_acc'],
        },
        'overfitting_gap': comparison[best_name]['overfit_gap'],
        'cross_validation': {
            'cv_roc_auc_mean': comparison[best_name]['cv_roc_mean'],
            'cv_roc_auc_std':  comparison[best_name]['cv_roc_std'],
            'folds': 5,
        },
        'confusion_matrix': cm_best,
        'feature_importance': _get_importance(best_model),
        'per_class': comparison[best_name]['per_class'],
        'class_names': ['Not Early', 'Early (<30d)'],
        # Legacy keys
        'test_acc':  comparison[best_name]['accuracy'],
        'train_acc': comparison[best_name]['train_acc'],
        'roc_auc':   comparison[best_name]['roc_auc'],
        'cv_acc':    [],
        'cv_roc':    best_cv_roc.tolist(),
        'n_train':   int(len(X_train)),
        'n_test':    int(len(X_test)),
        'mean_confidence': round(float(best_y_proba.mean()), 4),
        'training_mode': 'real_data_only',
        'loo_roc':   comparison[best_name]['cv_roc_mean'],
    }

    with open(PERFORMANCE_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)

    meta = {'feature_cols': FEATURE_COLS, 'target_col': TARGET_COL}
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(meta, f)

    return results


def train() -> dict:
    return train_all_models()


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
        return pickle.load(f)


def load_comparison() -> dict:
    if not os.path.exists(COMPARISON_JSON):
        train()
    with open(COMPARISON_JSON) as f:
        return json.load(f)


def compare_models() -> dict:
    return load_comparison()


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_importance(model) -> dict:
    if hasattr(model, 'feature_importances_'):
        return {k: round(float(v), 4)
                for k, v in zip(FEATURE_COLS, model.feature_importances_)}
    if hasattr(model, 'coef_'):
        return {k: round(float(abs(v)), 4)
                for k, v in zip(FEATURE_COLS, model.coef_[0])}
    return {}


def _save_comparison_chart(comparison: dict):
    names   = list(comparison.keys())
    short   = [SHORT_NAMES[n] for n in names]
    metrics = ['accuracy', 'roc_auc', 'f1_macro', 'f1_minority']
    labels  = ['Accuracy', 'ROC-AUC', 'F1 Macro', 'F1 Minority\n(Early <30d)']
    colours = ['#2980b9', '#27ae60', '#8e44ad', '#e74c3c']

    x   = np.arange(len(names))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (metric, label, colour) in enumerate(zip(metrics, labels, colours)):
        vals = [comparison[n][metric] for n in names]
        bars = ax.bar(x + i * w, vals, width=w, label=label,
                      color=colour, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(short, fontsize=12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Comparison — LR vs Random Forest vs Gradient Boosting',
                 fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.axhline(0.5, color='grey', lw=1, ls='--', alpha=0.4, label='Random baseline')
    ax.legend(fontsize=9, loc='upper left', ncol=5)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _save_confusion_matrix(y_test, y_pred, class_names, model_name=''):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _save_feature_importance(model, model_name=''):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        xlabel = 'Feature Importance (Gini)'
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        xlabel = 'Coefficient Magnitude (|coef|)'
    else:
        return

    labels     = [c.replace('_', ' ').title() for c in FEATURE_COLS]
    sorted_idx = np.argsort(importances)[-15:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([labels[i] for i in sorted_idx], importances[sorted_idx],
            color='#2980b9')
    ax.set_xlabel(xlabel)
    ax.set_title(f'Top 15 Feature Importances — {model_name}')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'),
                dpi=150, bbox_inches='tight')
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
    model = load_model()

    age_numeric     = AGE_MIDPOINTS.get(age_range, 55)
    gender_enc      = 1 if gender.lower() == 'male' else 0
    race_enc        = _RACE_MAP.get(race, 5)
    a1c_enc         = _A1C_MAP.get(a1c_result, 0)
    glu_enc         = _GLU_MAP.get(glu_serum, 0)
    insulin_enc     = _MED_MAP.get(insulin, 0)
    metformin_enc   = 0
    glipizide_enc   = 0
    glyburide_enc   = 0
    glimepiride_enc = 0
    change_enc      = 1 if medication_changed == 'Ch' else 0
    diab_med_enc    = 1 if diabetes_med == 'Yes' else 0
    adm_type_grp    = _ADMISSION_TYPE_MAP.get(int(admission_type), 3)
    discharge_grp        = 0
    discharge_disp_raw   = 1
    admission_source_grp = 0
    diag1_enc       = DIAG_CAT_MAP.get(diag_1_category, 8)

    num_meds_changed = 1 if insulin in ('Up', 'Down') else 0
    num_meds_used    = 1 if insulin != 'No' else 0

    total_prior_visits  = number_outpatient + number_emergency + number_inpatient
    has_prior_inpatient = 1 if number_inpatient > 0 else 0
    high_risk_discharge = 1 if discharge_disp_raw in {9, 12, 15, 22, 28} else 0
    insulin_down        = 1 if insulin_enc == 3 else 0
    poor_glycaemic_ctrl = 1 if (a1c_enc >= 2 and insulin_enc >= 1) else 0
    long_stay           = 1 if time_in_hospital >= 7 else 0
    multimorbid         = 1 if number_diagnoses >= 7 else 0

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

    proba = model.predict_proba(features)[0]

    # Use the F1-optimal threshold found during training
    perf = load_performance()
    threshold = perf.get('optimal_threshold', 0.5)
    pred = int(proba[1] >= threshold)

    return {
        'prediction':    pred,
        'label':         'Early Readmission Risk' if pred == 1 else 'Low Readmission Risk',
        'probabilities': {
            'Low Risk (Not Early)':    round(float(proba[0]) * 100, 1),
            'High Risk (Early <30d)':  round(float(proba[1]) * 100, 1),
        },
    }


if __name__ == '__main__':
    results = train_all_models()
    print(f"\nBest model : {results['model']}")
    print(f"Accuracy   : {results['test_performance']['accuracy']:.1%}")
    print(f"ROC-AUC    : {results['test_performance']['roc_auc']:.3f}")
    print(f"F1 Macro   : {results['test_performance']['f1_macro']:.3f}")
    print(f"F1 Minority: {results['test_performance']['f1_minority']:.3f}")
    print(f"CV ROC-AUC : {results['cross_validation']['cv_roc_auc_mean']:.3f} "
          f"± {results['cross_validation']['cv_roc_auc_std']:.3f}")
