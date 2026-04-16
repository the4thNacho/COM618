"""
Exploratory data analysis charts for the Diabetes 130-US Hospitals dataset.

All charts are saved to exploration_visuals/ and served via /image/<name>.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from cleaning import load_and_clean, FEATURE_COLS

EXPLORATION_DIR = os.path.join(os.path.dirname(__file__), 'exploration_visuals')


def _ensure_dir():
    os.makedirs(EXPLORATION_DIR, exist_ok=True)


def _load() -> pd.DataFrame:
    return load_and_clean()


# ─────────────────────────────────────────────────────────────────────────────
# Individual charts
# ─────────────────────────────────────────────────────────────────────────────

def generate_readmission_distribution() -> str:
    """Bar chart showing readmission class breakdown."""
    _ensure_dir()
    df = _load()

    counts = df['readmitted_early'].value_counts().sort_index()
    labels = ['Not Early\n(>30d or NO)', 'Early\n(<30 days)']
    colours = ['#2ecc71', '#e74c3c']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colours, edgecolor='black', width=0.5)
    ax.bar_label(bars, fmt='%d', padding=4, fontsize=11)
    ax.set_ylabel('Number of Encounters')
    ax.set_title('Readmission Class Distribution', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'diagnosis_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_correlation_heatmap() -> str:
    """Correlation heatmap of numerical features."""
    _ensure_dir()
    df = _load()

    num_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'age_numeric',
        'num_meds_changed', 'readmitted_early',
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 8})
    ax.set_title('Correlation Heatmap — Numerical Features', fontsize=14, pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'correlation_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_age_readmission() -> str:
    """Readmission rate by age group."""
    _ensure_dir()
    df = _load()

    age_labels = [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)',
    ]
    from cleaning import AGE_MIDPOINTS
    df['age_range'] = df['age_numeric'].map(
        {v: k for k, v in AGE_MIDPOINTS.items()}
    )
    rate = (
        df.groupby('age_range')['readmitted_early']
        .mean()
        .reindex(age_labels) * 100
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(rate.index, rate.values, color='#2980b9', edgecolor='black')
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Early Readmission Rate (%)')
    ax.set_title('Early Readmission Rate by Age Group', fontsize=14)
    ax.set_ylim(0, rate.max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'age_readmission.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_diag_readmission() -> str:
    """Readmission rate by primary diagnosis category."""
    _ensure_dir()
    df = _load()

    rate = (
        df.groupby('diag_1_category')['readmitted_early']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'rate', 'count': 'n'})
        .sort_values('rate', ascending=True)
    )
    rate['rate'] *= 100

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(rate.index, rate['rate'], color='#e67e22', edgecolor='black')
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=9)
    ax.set_xlabel('Early Readmission Rate (%)')
    ax.set_title('Early Readmission Rate by Primary Diagnosis Category', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'diag_readmission.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_medication_readmission() -> str:
    """Readmission rate by insulin usage category."""
    _ensure_dir()
    df = _load()

    rate = (
        df.groupby('insulin')['readmitted_early']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'rate', 'count': 'n'})
        .sort_values('rate', ascending=False)
    )
    rate['rate'] *= 100

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    bars = ax.bar(rate.index, rate['rate'], color=colours[:len(rate)], edgecolor='black')
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)
    ax.set_xlabel('Insulin Usage')
    ax.set_ylabel('Early Readmission Rate (%)')
    ax.set_title('Early Readmission Rate by Insulin Usage', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'medication_readmission.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_time_in_hospital() -> str:
    """Histogram of time in hospital split by readmission."""
    _ensure_dir()
    df = _load()

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, colour, val in [('Not Early', '#2ecc71', 0), ('Early (<30d)', '#e74c3c', 1)]:
        subset = df[df['readmitted_early'] == val]['time_in_hospital']
        ax.hist(subset, bins=range(1, 16), alpha=0.6, label=label,
                color=colour, edgecolor='black', density=True)

    ax.set_xlabel('Time in Hospital (days)')
    ax.set_ylabel('Density')
    ax.set_title('Time in Hospital Distribution by Readmission', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'time_in_hospital.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_race_readmission() -> str:
    """Readmission rate by race."""
    _ensure_dir()
    df = _load()

    rate = (
        df.groupby('race')['readmitted_early']
        .mean()
        .sort_values(ascending=True) * 100
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(rate.index, rate.values, color='#8e44ad', edgecolor='black')
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=10)
    ax.set_xlabel('Early Readmission Rate (%)')
    ax.set_title('Early Readmission Rate by Race', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'race_readmission.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_all() -> dict:
    """Generate all exploration charts and return their file paths."""
    return {
        'diagnosis_distribution':  generate_readmission_distribution(),
        'correlation_heatmap':     generate_correlation_heatmap(),
        'age_readmission':         generate_age_readmission(),
        'diag_readmission':        generate_diag_readmission(),
        'medication_readmission':  generate_medication_readmission(),
        'time_in_hospital':        generate_time_in_hospital(),
        'race_readmission':        generate_race_readmission(),
    }


if __name__ == '__main__':
    paths = generate_all()
    for name, path in paths.items():
        print(f'{name}: {path}')
