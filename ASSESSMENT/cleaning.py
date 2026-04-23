"""
Data cleaning and feature engineering for the Diabetes 130-US Hospitals dataset.

Source  : UCI ML Repository — Diabetes 130-US Hospitals for Years 1999-2008
Records : 101,766 patient encounters, 50 columns
Target  : Binary early readmission — within 30 days (<30) = 1, else = 0

Dirty data issues in this dataset:
  - '?' used throughout for missing values (not NaN)
  - 96.9 % of weight values missing — column dropped
  - 49 % of medical_specialty missing — column dropped
  - 39.6 % of payer_code missing — column dropped
  - 83–95 % of A1Cresult / max_glu_serum missing — kept as 'None' category
  - Age stored as string ranges ('[60-70)') not numbers
  - Medications stored as strings (No / Steady / Up / Down)
  - Multiple encounters per patient (data leakage risk)
  - 1,652 deceased patients who cannot logically be readmitted
  - 'Unknown/Invalid' gender entries
  - ICD-9 codes are raw strings needing grouping

All major decisions are documented in docs/dataset_decisions.md
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = os.path.join(
    os.path.dirname(__file__),
    'diabetes+130-us+hospitals+for+years+1999-2008',
    'diabetic_data.csv',
)
CLEANED_PATH     = os.path.join(os.path.dirname(__file__), 'diabetes_cleaned.csv')
CLEANING_VISUALS = os.path.join(os.path.dirname(__file__), 'cleaning_visuals')

# Medication columns: No / Steady / Up / Down
MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
]

# Patients discharged as deceased — cannot be readmitted
DECEASED_DISCHARGE_IDS = {11, 19, 20}

# Age range → numeric midpoint
AGE_MIDPOINTS = {
    '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95,
}

# Encoding maps
_A1C_MAP  = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
_GLU_MAP  = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
_MED_MAP  = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
_RACE_MAP = {
    'Caucasian': 0, 'AfricanAmerican': 1, 'Hispanic': 2,
    'Asian': 3, 'Other': 4, 'Unknown': 5,
}
_ADMISSION_TYPE_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}

_DIAG_CAT_ORDER = [
    'Diabetes', 'Circulatory', 'Respiratory', 'Digestive',
    'Genitourinary', 'Musculoskeletal', 'Injury', 'Neoplasms', 'Other',
]
DIAG_CAT_MAP = {cat: i for i, cat in enumerate(_DIAG_CAT_ORDER)}


def _icd9_category(code) -> str:
    """Map an ICD-9 code string to a broad disease category."""
    if pd.isna(code) or str(code).strip() == '':
        return 'Other'
    code = str(code).strip()
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    try:
        val = float(code)
    except ValueError:
        return 'Other'
    if 250 <= val < 251:       return 'Diabetes'
    if (390 <= val < 460) or val == 785: return 'Circulatory'
    if (460 <= val < 520) or val == 786: return 'Respiratory'
    if (520 <= val < 580) or val == 787: return 'Digestive'
    if (580 <= val < 630) or val == 788: return 'Genitourinary'
    if 710 <= val < 740:       return 'Musculoskeletal'
    if 800 <= val < 1000:      return 'Injury'
    if 140 <= val < 240:       return 'Neoplasms'
    return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Load the raw dataset with '?' treated as NaN."""
    return pd.read_csv(RAW_PATH, na_values=['?'], low_memory=False)


def get_dirty_stats(raw: pd.DataFrame) -> dict:
    """Compute before-cleaning statistics for display on the cleaning page."""
    missing = (raw.isnull().sum() / len(raw) * 100).round(1).to_dict()
    return {
        'total_rows':       len(raw),
        'total_cols':       len(raw.columns),
        'total_missing_pct': round(raw.isnull().mean().mean() * 100, 1),
        'missing_by_col':   {k: v for k, v in missing.items() if v > 0},
        'question_mark_cols': [
            'weight', 'payer_code', 'medical_specialty', 'race',
            'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
        ],
        'deceased_patients': int(raw['discharge_disposition_id'].isin(DECEASED_DISCHARGE_IDS).sum()),
        'duplicate_patients': int(raw.duplicated('patient_nbr').sum()),
        'invalid_gender': int((raw['gender'] == 'Unknown/Invalid').sum()),
    }


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning and feature-engineering steps.

    Returns a fully cleaned, encoded DataFrame ready for ML.
    """
    df = raw.copy()

    # ── Step 1: Drop rows for deceased patients ───────────────────────────────
    # Reason: deceased patients cannot be readmitted; including them would
    # teach the model spurious patterns (they have readmitted=NO by definition).
    df = df[~df['discharge_disposition_id'].isin(DECEASED_DISCHARGE_IDS)].copy()

    # ── Step 2: Remove duplicate patient encounters ───────────────────────────
    # Keep only each patient's first encounter.
    # Reason: multiple encounters per patient leak information across train/test
    # split boundaries and violate the i.i.d. assumption.
    df = df.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first').copy()

    # ── Step 3: Drop high-missingness / low-signal columns ───────────────────
    # weight          — 96.9 % missing, no meaningful imputation possible
    # payer_code      — 39.6 % missing, insurance type not clinically actionable
    # medical_specialty — 49 % missing, admission_type_id carries similar signal
    df = df.drop(columns=['weight', 'payer_code', 'medical_specialty'], errors='ignore')

    # ── Step 4: Drop identifier columns ───────────────────────────────────────
    df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')

    # ── Step 5: Fix invalid gender ────────────────────────────────────────────
    # 'Unknown/Invalid' is treated as Female (mode) since only 3 records affected.
    df['gender'] = df['gender'].replace('Unknown/Invalid', 'Female')

    # ── Step 6: Fill remaining categorical missing values ─────────────────────
    df['race'] = df['race'].fillna('Unknown')
    df['A1Cresult']    = df['A1Cresult'].fillna('None')
    df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
    df['diag_1'] = df['diag_1'].fillna('Other')
    df['diag_2'] = df['diag_2'].fillna('Other')
    df['diag_3'] = df['diag_3'].fillna('Other')

    # ── Step 7: Encode age range → numeric midpoint ───────────────────────────
    df['age_numeric'] = df['age'].map(AGE_MIDPOINTS).astype(float)

    # ── Step 8: Encode gender and race ───────────────────────────────────────
    df['gender_enc'] = (df['gender'] == 'Male').astype(int)
    df['race_enc']   = df['race'].map(_RACE_MAP).fillna(5).astype(int)

    # ── Step 9: Encode lab results ────────────────────────────────────────────
    df['a1c_result_enc']  = df['A1Cresult'].map(_A1C_MAP).fillna(0).astype(int)
    df['glu_serum_enc']   = df['max_glu_serum'].map(_GLU_MAP).fillna(0).astype(int)

    # ── Step 10: Encode medication columns ────────────────────────────────────
    for col in MED_COLS:
        df[col] = df[col].fillna('No')
        df[f'{col}_enc'] = df[col].map(_MED_MAP).fillna(0).astype(int)

    # ── Step 11: Engineer medication summary features ─────────────────────────
    df['num_meds_changed'] = sum(
        (df[f'{c}_enc'] == 2) | (df[f'{c}_enc'] == 3) for c in MED_COLS
    )
    df['num_meds_used'] = sum(df[f'{c}_enc'] >= 1 for c in MED_COLS)

    # ── Step 12: Group admission type ─────────────────────────────────────────
    df['admission_type_grp'] = df['admission_type_id'].map(
        _ADMISSION_TYPE_MAP
    ).fillna(3).astype(int)

    # ── Step 12a: Ensure discharge_disposition_id is clean numeric ───────────
    df['discharge_disposition_id'] = pd.to_numeric(
        df['discharge_disposition_id'], errors='coerce'
    ).fillna(1).astype(int)

    # ── Step 12b: Group discharge disposition ─────────────────────────────────
    # Groups: 0=Home, 1=Transfer/SNF, 2=Home with services, 3=AMA/Other
    # discharge_disposition_id is available at prediction time (post-discharge).
    def _discharge_grp(did):
        if pd.isna(did):
            return 3
        did = int(did)
        if did == 1:
            return 0   # home
        if did in (2, 3, 4, 5, 10):
            return 1   # transfer / institutional
        if did in (6, 8):
            return 2   # home with services
        return 3       # other / AMA / hospice
    df['discharge_grp'] = df['discharge_disposition_id'].apply(_discharge_grp).astype(int)

    # ── Step 12c: Group admission source ──────────────────────────────────────
    # 0=Physician referral, 1=Emergency, 2=Transfer, 3=Other
    def _source_grp(sid):
        if pd.isna(sid):
            return 3
        sid = int(sid)
        if sid == 1:
            return 0   # physician referral
        if sid in (7,):
            return 1   # emergency room
        if sid in (4, 5, 6):
            return 2   # transfer from hospital
        return 3
    df['admission_source_grp'] = df['admission_source_id'].apply(_source_grp).astype(int)

    # ── Step 13: ICD-9 diagnosis category encoding ────────────────────────────
    for diag_col in ('diag_1', 'diag_2', 'diag_3'):
        cat_col = f'{diag_col}_category'
        df[cat_col]         = df[diag_col].apply(_icd9_category)
        df[f'{diag_col}_cat_enc'] = df[cat_col].map(DIAG_CAT_MAP).fillna(8).astype(int)

    # ── Step 14: Binary flags ─────────────────────────────────────────────────
    df['change_enc']       = (df['change'] == 'Ch').astype(int)
    df['diabetes_med_enc'] = (df['diabetesMed'] == 'Yes').astype(int)

    # ── Step 14b: High-signal interaction / derived features ──────────────────
    # Prior care intensity — combined utilisation score
    df['total_prior_visits'] = (
        df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    )
    df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)

    # High-risk discharge codes (readmission rate ≥25% in EDA)
    HIGH_RISK_DISCHARGE = {9, 12, 15, 22, 28}
    df['high_risk_discharge'] = df['discharge_disposition_id'].isin(
        HIGH_RISK_DISCHARGE
    ).astype(int)

    # Insulin dose decreased — strong readmission signal from EDA
    df['insulin_down'] = (df['insulin_enc'] == 3).astype(int)

    # Poorly controlled diabetes: high A1C AND on insulin
    df['poor_glycaemic_control'] = (
        (df['a1c_result_enc'] >= 2) & (df['insulin_enc'] >= 1)
    ).astype(int)

    # Long-stay indicator (>= 7 days correlates with complexity)
    df['long_stay'] = (df['time_in_hospital'] >= 7).astype(int)

    # Many diagnoses — indicator of multimorbidity
    df['multimorbid'] = (df['number_diagnoses'] >= 7).astype(int)

    # ── Step 15: Target variable ──────────────────────────────────────────────
    # Binary: readmitted within 30 days = 1 (early readmission), else = 0.
    # This is the clinically significant outcome — avoiding early readmission
    # within 30 days is the primary focus of hospital readmission reduction programmes.
    df['readmitted_early'] = (df['readmitted'] == '<30').astype(int)

    return df


def run_cleaning_pipeline(force: bool = False) -> pd.DataFrame:
    """
    Run the full cleaning pipeline, saving diabetes_cleaned.csv.

    Uses cached file if it exists (pass force=True to regenerate).
    """
    if not force and os.path.exists(CLEANED_PATH):
        return pd.read_csv(CLEANED_PATH)

    raw = load_raw()
    cleaned = clean(raw)
    cleaned.to_csv(CLEANED_PATH, index=False)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning visualisations
# ─────────────────────────────────────────────────────────────────────────────

def generate_missing_values_chart() -> str:
    """Bar chart of missing value percentages before cleaning."""
    os.makedirs(CLEANING_VISUALS, exist_ok=True)
    raw = load_raw()

    missing = (raw.isnull().sum() / len(raw) * 100)
    missing = missing[missing > 0].sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ['#e74c3c' if v > 30 else '#e67e22' if v > 5 else '#f1c40f'
               for v in missing.values]
    bars = ax.barh(missing.index, missing.values, color=colours, edgecolor='black')
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=9)
    ax.set_xlabel('Missing Values (%)')
    ax.set_title('Missing Value Rate by Column (Before Cleaning)', fontsize=13)
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='30% threshold (drop)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()

    out = os.path.join(CLEANING_VISUALS, 'missing_values.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_readmission_class_chart() -> str:
    """Pie chart of readmission class distribution after cleaning."""
    os.makedirs(CLEANING_VISUALS, exist_ok=True)
    df = run_cleaning_pipeline()

    counts = df['readmitted'].value_counts()
    colours = ['#2ecc71', '#3498db', '#e74c3c']
    labels = [f'{k}\n({v:,})' for k, v in counts.items()]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colours,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax.set_title('Readmission Class Distribution (Cleaned Dataset)', fontsize=13)
    plt.tight_layout()

    out = os.path.join(CLEANING_VISUALS, 'readmission_classes.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_before_after_chart() -> str:
    """Bar chart comparing row count before vs after cleaning."""
    os.makedirs(CLEANING_VISUALS, exist_ok=True)
    raw     = load_raw()
    cleaned = run_cleaning_pipeline()

    stages = ['Raw Dataset', 'Remove Deceased', 'Deduplicate\n(1 per patient)', 'Final Cleaned']
    n_deceased = raw[raw['discharge_disposition_id'].isin(DECEASED_DISCHARGE_IDS)].shape[0]
    n_after_deceased = len(raw) - n_deceased
    n_dedup = len(raw.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first'))

    values = [len(raw), n_after_deceased, n_dedup, len(cleaned)]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
    bars = ax.bar(stages, values, color=colours, edgecolor='black', width=0.5)
    ax.bar_label(bars, fmt='%d', padding=4, fontsize=10)
    ax.set_ylabel('Number of Records')
    ax.set_title('Record Count at Each Cleaning Stage', fontsize=13)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(CLEANING_VISUALS, 'before_after.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_before_after_columns_chart() -> str:
    """
    2×2 grid showing four dirty→clean column transformations:
      1. Age: string ranges → numeric histogram
      2. A1Cresult: raw strings → ordinal 0-3
      3. Primary diagnosis: 700+ ICD-9 codes → 9 categories
      4. Insulin: string → ordinal with readmission rate overlay
    """
    os.makedirs(CLEANING_VISUALS, exist_ok=True)
    raw     = load_raw()
    cleaned = run_cleaning_pipeline()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Before vs After Cleaning — Column Transformations', fontsize=14, y=1.01)

    # ── Panel 1: Age ────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    age_order = list(AGE_MIDPOINTS.keys())
    age_midpoints = list(AGE_MIDPOINTS.values())
    age_counts = raw['age'].value_counts().reindex(age_order, fill_value=0)
    ax2 = ax.twinx()
    ax.bar(age_midpoints, age_counts.values, width=8, color='#e74c3c', alpha=0.7, label='Raw (string)')
    ax2.hist(cleaned['age_numeric'], bins=10, color='#2980b9', alpha=0.6, label='Clean (numeric)')
    ax.set_xticks(age_midpoints)
    ax.set_xticklabels(age_order, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Age bracket', fontsize=9)
    ax.set_ylabel('Count (raw)', color='#e74c3c')
    ax2.set_ylabel('Count (cleaned)', color='#2980b9')
    ax.set_title('Age: "[60-70)" → 65 (midpoint)', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    # ── Panel 2: A1Cresult ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    raw_a1c = raw['A1Cresult'].fillna('None').value_counts()
    clean_a1c = cleaned['a1c_result_enc'].value_counts().sort_index()
    x_pos = [0, 1, 2, 3]
    enc_labels = ['None (0)', 'Norm (1)', '>7 (2)', '>8 (3)']
    # raw bar
    raw_vals = [raw_a1c.get(k, 0) for k in ['None', 'Norm', '>7', '>8']]
    ax_r = ax.twinx()
    ax.bar([p - 0.2 for p in x_pos], raw_vals, width=0.35,
           color='#e74c3c', alpha=0.7, label='Raw (string)')
    ax_r.bar([p + 0.2 for p in x_pos],
             [clean_a1c.get(i, 0) for i in range(4)], width=0.35,
             color='#2980b9', alpha=0.7, label='Clean (ordinal 0-3)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(enc_labels, fontsize=9)
    ax.set_ylabel('Count (raw)', color='#e74c3c')
    ax_r.set_ylabel('Count (clean)', color='#2980b9')
    ax.set_title('A1Cresult: missing/"?" → ordinal 0–3', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    # ── Panel 3: ICD-9 diagnosis code reduction ──────────────────────────────────
    ax = axes[1, 0]
    raw_diag = raw['diag_1'].dropna()
    n_raw_unique = raw_diag.nunique()
    clean_diag = cleaned['diag_1_category'].value_counts()
    order = ['Diabetes', 'Circulatory', 'Respiratory', 'Digestive',
             'Genitourinary', 'Musculoskeletal', 'Injury', 'Neoplasms', 'Other']
    vals = [clean_diag.get(c, 0) for c in order]
    colours = plt.cm.Set3(np.linspace(0, 1, 9))
    bars = ax.barh(order, vals, color=colours, edgecolor='black')
    ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)
    ax.set_title(f'ICD-9 diag_1: {n_raw_unique} unique codes → 9 categories', fontsize=11)
    ax.set_xlabel('Patient Count')
    ax.grid(axis='x', alpha=0.3)

    # ── Panel 4: Insulin encoding with readmission rate ─────────────────────────
    ax = axes[1, 1]
    insulin_vals = ['No', 'Steady', 'Up', 'Down']
    enc_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
    raw_counts = [raw['insulin'].fillna('No').value_counts().get(v, 0) for v in insulin_vals]
    clean_rates = []
    for v in insulin_vals:
        enc = enc_map[v]
        mask = cleaned['insulin_enc'] == enc
        rate = cleaned.loc[mask, 'readmitted_early'].mean() * 100 if mask.sum() > 0 else 0
        clean_rates.append(round(rate, 1))
    colours_ins = ['#95a5a6', '#3498db', '#27ae60', '#e74c3c']
    ax_r2 = ax.twinx()
    ax.bar(insulin_vals, raw_counts, color=colours_ins, alpha=0.65, label='Patient count (raw)')
    ax_r2.plot(insulin_vals, clean_rates, 'ko-', lw=2, ms=8, label='Readmission rate %')
    ax.set_ylabel('Patient Count (raw)', fontsize=10)
    ax_r2.set_ylabel('Early Readmission Rate (%)', fontsize=10)
    ax.set_title('Insulin: string → ordinal 0-3 (with readmission rate)', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    plt.tight_layout()
    out = os.path.join(CLEANING_VISUALS, 'before_after_columns.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_all_cleaning_charts() -> dict:
    """Generate all cleaning-stage charts."""
    return {
        'missing_values':       generate_missing_values_chart(),
        'readmission_classes':  generate_readmission_class_chart(),
        'before_after':         generate_before_after_chart(),
        'before_after_columns': generate_before_after_columns_chart(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cleaned data interface for predictor.py
# ─────────────────────────────────────────────────────────────────────────────

# Exact feature columns used by the model
FEATURE_COLS = [
    # Demographics
    'age_numeric',
    'gender_enc',
    'race_enc',
    # Admission context
    'time_in_hospital',
    'admission_type_grp',
    'discharge_grp',
    'discharge_disposition_id',
    'admission_source_grp',
    # Clinical volume
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
    # Lab results
    'a1c_result_enc',
    'glu_serum_enc',
    # Medications
    'insulin_enc',
    'metformin_enc',
    'glipizide_enc',
    'glyburide_enc',
    'glimepiride_enc',
    'change_enc',
    'diabetes_med_enc',
    'num_meds_changed',
    'num_meds_used',
    # Engineered interaction / derived features
    'total_prior_visits',
    'has_prior_inpatient',
    'high_risk_discharge',
    'insulin_down',
    'poor_glycaemic_control',
    'long_stay',
    'multimorbid',
    # Diagnosis categories
    'diag_1_cat_enc',
    'diag_2_cat_enc',
    'diag_3_cat_enc',
]

TARGET_COL = 'readmitted_early'


def load_and_clean() -> pd.DataFrame:
    """Return the cleaned DataFrame (uses cache if available)."""
    return run_cleaning_pipeline()


def get_features_and_target(df: pd.DataFrame | None = None):
    """Return (X, y) ready for sklearn."""
    if df is None:
        df = load_and_clean()
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def get_cleaning_stats() -> dict:
    """Return summary statistics for the home page and cleaning page."""
    raw     = load_raw()
    dirty   = get_dirty_stats(raw)
    cleaned = run_cleaning_pipeline()
    rc = cleaned['readmitted_early'].value_counts()
    return {
        'raw_rows':             dirty['total_rows'],
        'cleaned_rows':         len(cleaned),
        'removed_rows':         dirty['total_rows'] - len(cleaned),
        'features':             len(FEATURE_COLS),
        'early_readmitted':     int(rc.get(1, 0)),
        'not_early_readmitted': int(rc.get(0, 0)),
        'readmission_rate':     round(int(rc.get(1, 0)) / len(cleaned) * 100, 1),
        'dirty':                dirty,
    }
