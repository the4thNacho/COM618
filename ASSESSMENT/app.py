"""
Flask application for COM618 Medical Data Analysis Assessment.

Dataset: Diabetes 130-US Hospitals for Years 1999-2008
Target : Predict early hospital readmission (within 30 days)

Routes:
    /           — Home page with dataset overview stats
    /cleaning   — Data cleaning pipeline and visualisations
    /exploration — EDA charts (patterns & trends)
    /model      — Predictive model + live prediction form
    /performance — Model performance dashboard
    /image/<n>  — Serve chart images from various output directories
"""

import os
import threading

import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify, url_for

from predictor import (
    train, predict as make_prediction,
    load_performance, load_comparison,
    MODEL_DIR, MODEL_PATH, PERFORMANCE_JSON, COMPARISON_JSON,
)
from cleaning import (
    get_cleaning_stats, run_cleaning_pipeline,
    generate_all_cleaning_charts, CLEANING_VISUALS,
    AGE_MIDPOINTS, DIAG_CAT_MAP,
)

app = Flask(__name__)

# ─── Background training state ───────────────────────────────────────────────
_training_lock  = threading.Lock()
_training_state = {'running': False, 'message': 'Idle', 'error': None}


def _needs_training() -> bool:
    return not (os.path.exists(PERFORMANCE_JSON) and os.path.exists(COMPARISON_JSON))


def _start_training_if_needed() -> bool:
    """Start background training if models are missing. Returns True if training was started or is running."""
    with _training_lock:
        if _training_state['running']:
            return True
        if not _needs_training():
            return False
        _training_state['running'] = True
        _training_state['message'] = 'Starting…'
        _training_state['error']   = None

    def _run():
        try:
            def cb(msg):
                with _training_lock:
                    _training_state['message'] = msg
            train(progress_cb=cb)
        except Exception as exc:
            with _training_lock:
                _training_state['error']   = str(exc)
                _training_state['message'] = f'Error: {exc}'
        finally:
            with _training_lock:
                _training_state['running'] = False

    threading.Thread(target=_run, daemon=True).start()
    return True


BASE_DIR            = os.path.dirname(__file__)
EXPLORATION_VISUALS = os.path.join(BASE_DIR, 'exploration_visuals')
PERFORMANCE_CHARTS  = os.path.join(MODEL_DIR, 'performance_charts')

IMAGE_DIRS = [
    CLEANING_VISUALS,
    os.path.join(CLEANING_VISUALS, 'dashboard'),
    EXPLORATION_VISUALS,
    MODEL_DIR,
    PERFORMANCE_CHARTS,
]

# ─── Cleaning strategy table ──────────────────────────────────────────────────
STRATEGY_TABLE = [
    {
        'col': 'weight',
        'issue': 'Missing',
        'missing_pct': '96.9%',
        'strategy': 'Drop Column',
        'justification': '97% of values are "?" — no meaningful imputation possible for such a sparse feature.',
    },
    {
        'col': 'payer_code',
        'issue': 'Missing',
        'missing_pct': '39.6%',
        'strategy': 'Drop Column',
        'justification': 'Insurance type is not clinically actionable for readmission prediction and has high missingness.',
    },
    {
        'col': 'medical_specialty',
        'issue': 'Missing',
        'missing_pct': '49.1%',
        'strategy': 'Drop Column',
        'justification': 'Nearly half missing; admission_type_id carries overlapping signal with less missingness.',
    },
    {
        'col': 'A1Cresult',
        'issue': 'Missing',
        'missing_pct': '83.3%',
        'strategy': 'Fill as "None" category',
        'justification': 'Missing means test was not performed — "None" is semantically correct and clinically informative.',
    },
    {
        'col': 'max_glu_serum',
        'issue': 'Missing',
        'missing_pct': '94.7%',
        'strategy': 'Fill as "None" category',
        'justification': 'Same rationale as A1Cresult — absence of a glucose test is itself a clinical signal.',
    },
    {
        'col': 'race',
        'issue': 'Missing ("?")',
        'missing_pct': '2.2%',
        'strategy': 'Fill as "Unknown"',
        'justification': 'Small proportion; "Unknown" preserves the row without fabricating race information.',
    },
    {
        'col': 'diag_1/2/3',
        'issue': 'Raw ICD-9 codes',
        'missing_pct': '0–1.4%',
        'strategy': 'Group into 9 disease categories',
        'justification': 'Over 700 unique codes — grouping into categories (Diabetes, Circulatory, etc.) provides generalizable signal.',
    },
    {
        'col': 'age',
        'issue': 'String range ("[60-70)")',
        'missing_pct': '0%',
        'strategy': 'Convert to numeric midpoint',
        'justification': 'Models require numerical input. Midpoint (65) is the best single estimate for an age range.',
    },
    {
        'col': 'gender (Unknown/Invalid)',
        'issue': 'Invalid category',
        'missing_pct': '<0.01%',
        'strategy': 'Replace with "Female" (mode)',
        'justification': 'Only 3 records affected — replacing with mode has negligible impact.',
    },
    {
        'col': 'Deceased patients',
        'issue': 'Logical impossibility',
        'missing_pct': '—',
        'strategy': 'Remove rows (1,652 records)',
        'justification': 'Patients who died during admission cannot be readmitted. Including them would introduce noise.',
    },
    {
        'col': 'Duplicate patient encounters',
        'issue': 'Data leakage risk',
        'missing_pct': '—',
        'strategy': 'Keep first encounter per patient',
        'justification': 'Multiple encounters per patient violate the i.i.d. assumption and risk leaking information across train/test splits.',
    },
    {
        'col': 'Medication columns (23)',
        'issue': 'String (No/Steady/Up/Down)',
        'missing_pct': '0%',
        'strategy': 'Ordinal encode + engineer summary features',
        'justification': 'Ordinal encoding preserves dosage direction. Summary features (num_meds_changed) capture overall medication management intensity.',
    },
]


def _get_metrics():
    return load_performance()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    stats = get_cleaning_stats()
    return render_template('index.html', stats=stats)


@app.route('/cleaning')
def cleaning():
    generate_all_cleaning_charts()
    stats = get_cleaning_stats()
    return render_template('cleaning.html',
                           strategy_table=STRATEGY_TABLE,
                           stats=stats)


@app.route('/exploration')
def exploration():
    from exploration import generate_all
    generate_all()
    stats = get_cleaning_stats()
    return render_template('exploration.html', readmission_rate=stats['readmission_rate'])


@app.route('/api/training_status')
def training_status():
    with _training_lock:
        state = dict(_training_state)
    state['ready'] = not _needs_training() and not state['running']
    return jsonify(state)


@app.route('/model', methods=['GET', 'POST'])
def model():
    if _start_training_if_needed():
        return render_template('training_loading.html', redirect_to=url_for('model'))
    metrics = _get_metrics()
    result  = None
    form    = {}

    age_ranges = list(AGE_MIDPOINTS.keys())
    diag_cats  = list(DIAG_CAT_MAP.keys())

    if request.method == 'POST':
        try:
            form = {k: request.form[k] for k in request.form}
            result = make_prediction(
                age_range         = form['age_range'],
                gender            = form['gender'],
                race              = form['race'],
                time_in_hospital  = int(form['time_in_hospital']),
                num_lab_procedures= int(form['num_lab_procedures']),
                num_procedures    = int(form['num_procedures']),
                num_medications   = int(form['num_medications']),
                number_outpatient = int(form['number_outpatient']),
                number_emergency  = int(form['number_emergency']),
                number_inpatient  = int(form['number_inpatient']),
                number_diagnoses  = int(form['number_diagnoses']),
                a1c_result        = form['a1c_result'],
                glu_serum         = form['glu_serum'],
                insulin           = form['insulin'],
                diabetes_med      = form['diabetes_med'],
                medication_changed= form['medication_changed'],
                admission_type    = int(form['admission_type']),
                diag_1_category   = form['diag_1_category'],
            )
        except (ValueError, KeyError) as e:
            result = {'error': str(e)}

    cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    _total = sum(cm[0]) + sum(cm[1])
    _pos   = sum(cm[1])
    minority_pct = round(_pos / _total * 100, 1) if _total else 0

    comparison = load_comparison()

    return render_template('model.html',
                           metrics=metrics, result=result, form=form,
                           age_ranges=age_ranges, diag_cats=diag_cats,
                           minority_pct=minority_pct,
                           comparison=comparison)


@app.route('/performance')
def performance():
    if _start_training_if_needed():
        return render_template('training_loading.html', redirect_to=url_for('performance'))
    from performance_dashboard import generate_all
    generate_all()
    perf = load_performance()
    import numpy as np

    # Derive majority/minority class rates from the confusion matrix so nothing
    # is hardcoded — these will update automatically if the model is retrained.
    cm = perf.get('confusion_matrix', [[0, 0], [0, 0]])
    total_test  = sum(cm[0]) + sum(cm[1])
    n_negative  = sum(cm[0])   # actual Not Early
    n_positive  = sum(cm[1])   # actual Early
    majority_pct  = round(n_negative / total_test * 100, 1) if total_test else 0
    minority_pct  = round(n_positive / total_test * 100, 1) if total_test else 0
    minority_weight = round(n_negative / n_positive, 1) if n_positive else 0

    cv_roc = perf.get('cv_roc', [0])
    cv_acc = perf.get('cv_acc', [0])

    # Build human-readable rows for the model config table from the stored config dict.
    # Descriptions are generic enough to apply to any GBM-family model.
    _cfg = perf.get('model_config', {})
    config_rows = []
    _desc = {
        'n_estimators':     'number of sequential trees',
        'learning_rate':    'step size per tree — smaller = less overfitting',
        'max_depth':        'max tree depth — limits interaction complexity',
        'min_samples_leaf': 'min patients per leaf — prevents noisy splits',
        'subsample':        'fraction of training rows used per tree',
        'max_features':     'feature subsampling at each split',
        'random_state':     'random seed for reproducibility',
    }
    for k, v in _cfg.items():
        config_rows.append({'param': k, 'value': v, 'desc': _desc.get(k, '')})

    stats = {
        'test_acc':          round(perf.get('test_acc', 0) * 100, 1),
        'train_acc':         round(perf.get('train_acc', 0) * 100, 1),
        'overfitting_gap':   perf.get('overfitting_gap', 0),
        'roc_auc':           round(perf.get('roc_auc', 0) * 100, 1),
        'cv_acc_mean':       round(float(np.mean(cv_acc)) * 100, 1),
        'cv_acc_std':        round(float(np.std(cv_acc))  * 100, 2),
        'cv_roc_mean':       round(float(np.mean(cv_roc)) * 100, 1),
        'cv_roc_std':        round(float(np.std(cv_roc))  * 100, 2),
        'mean_confidence':   perf.get('mean_confidence', 0),
        'per_class':         perf.get('per_class', {}),
        'class_names':       perf.get('class_names', []),
        'overfit_warning':   abs(perf.get('overfitting_gap', 0)) > 10,
        'best_model':        perf.get('model', 'GradientBoostingClassifier'),
        'training_mode':     perf.get('training_mode', 'real_data_only'),
        'n_train':           perf.get('n_train', 0),
        'n_test':            perf.get('n_test', 0),
        'avg_precision':     round(perf.get('test_performance', {}).get('avg_precision', 0) * 100, 1),
        # Derived class-balance stats — all from confusion matrix, nothing hardcoded
        'majority_pct':      majority_pct,
        'minority_pct':      minority_pct,
        'minority_weight':   minority_weight,
        # Model config as a list of rows for the config table
        'config_rows':       config_rows,
    }
    return render_template('performance.html', stats=stats)


@app.route('/image/<name>')
def image(name):
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return send_file(path, mimetype='image/png')
    return f'Image not found: {name}', 404


@app.route('/api/cluster/elbow')
def api_cluster_elbow():
    from clustering import generate_elbow_chart
    b64 = generate_elbow_chart()
    return jsonify({'elbow': b64})


@app.route('/api/cluster')
def api_cluster():
    from clustering import generate_cluster_results
    k = max(2, min(8, int(request.args.get('k', 3))))
    pca_b64, profile_b64, metrics = generate_cluster_results(k)
    return jsonify({'pca': pca_b64, 'profile': profile_b64, 'metrics': metrics})


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        train()
    app.run(debug=True, port=5000)
