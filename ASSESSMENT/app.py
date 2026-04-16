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

import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify

from predictor import (
    train, predict as make_prediction,
    load_performance, load_comparison,
    MODEL_DIR, MODEL_PATH,
)
from cleaning import (
    get_cleaning_stats, run_cleaning_pipeline,
    generate_all_cleaning_charts, CLEANING_VISUALS,
    AGE_MIDPOINTS, DIAG_CAT_MAP,
)

app = Flask(__name__)

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
    try:
        return load_performance()
    except Exception:
        return train()


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
    return render_template('exploration.html')


@app.route('/model', methods=['GET', 'POST'])
def model():
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

    return render_template('model.html',
                           metrics=metrics, result=result, form=form,
                           age_ranges=age_ranges, diag_cats=diag_cats)


@app.route('/performance')
def performance():
    from performance_dashboard import generate_all
    generate_all()
    perf = load_performance()
    import numpy as np

    stats = {
        'test_acc':        round(perf.get('test_acc', 0) * 100, 1),
        'train_acc':       round(perf.get('train_acc', 0) * 100, 1),
        'overfitting_gap': perf.get('overfitting_gap', 0),
        'roc_auc':         round(perf.get('roc_auc', 0) * 100, 1),
        'cv_acc_mean':     round(float(np.mean(perf.get('cv_acc', [0]))) * 100, 1),
        'cv_roc_mean':     round(float(np.mean(perf.get('cv_roc', [0]))) * 100, 1),
        'loo_acc':         round(perf.get('loo_acc', 0) * 100, 1),
        'loo_roc':         round(perf.get('loo_roc', 0) * 100, 1),
        'mean_confidence': perf.get('mean_confidence', 0),
        'per_class':       perf.get('per_class', {}),
        'class_names':     perf.get('class_names', []),
        'overfit_warning': abs(perf.get('overfitting_gap', 0)) > 10,
        'best_model':      'GradientBoostingClassifier',
        'comparison':      {},
        'training_mode':   perf.get('training_mode', 'real_data_only'),
        'n_train':         perf.get('n_train', 0),
        'n_test':          perf.get('n_test', 0),
        'avg_precision':   round(perf.get('test_performance', {}).get('avg_precision', 0) * 100, 1),
    }
    return render_template('performance.html', stats=stats)


@app.route('/image/<name>')
def image(name):
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return send_file(path, mimetype='image/png')
    return f'Image not found: {name}', 404


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        train()
    app.run(debug=True, port=5000)
