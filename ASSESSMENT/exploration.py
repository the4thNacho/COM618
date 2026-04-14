import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

CLEANED_PATH = os.path.join(os.path.dirname(__file__), 'realworld_medical_dirty_cleaned_1.csv')
EXPLORATION_DIR = os.path.join(os.path.dirname(__file__), 'exploration_visuals')


def load_data() -> pd.DataFrame:
    return pd.read_csv(CLEANED_PATH, keep_default_na=False, na_values=[''])


def ensure_dir():
    os.makedirs(EXPLORATION_DIR, exist_ok=True)


def generate_correlation_heatmap() -> str:
    """Correlation heatmap of numerical features."""
    ensure_dir()
    df = load_data()
    numerical = df[['Age', 'Blood_Pressure', 'Cholesterol', 'BMI']].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = numerical.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Correlation Between Numerical Features', fontsize=14, pad=12)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'correlation_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_diagnosis_distribution() -> str:
    """Bar chart of diagnosis counts."""
    ensure_dir()
    df = load_data()

    counts = df['Diagnosis'].value_counts()
    colours = ['#3498db', '#e74c3c', '#95a5a6']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=colours[:len(counts)], edgecolor='black')
    ax.bar_label(bars, padding=3)
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Patient Diagnosis Distribution', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'diagnosis_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_feature_boxplots() -> str:
    """Box plots of numerical features grouped by diagnosis (excluding UNKNOWN)."""
    ensure_dir()
    df = load_data()
    df_known = df[df['Diagnosis'] != 'UNKNOWN'].copy()

    features = ['Age', 'Blood_Pressure', 'Cholesterol', 'BMI']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    palette = {'DIABETES': '#3498db', 'HEART DISEASE': '#e74c3c'}

    for i, feat in enumerate(features):
        sns.boxplot(data=df_known, x='Diagnosis', y=feat, hue='Diagnosis',
                    palette=palette, legend=False, ax=axes[i])
        axes[i].set_title(f'{feat} by Diagnosis')
        axes[i].set_xlabel('')
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Distributions by Diagnosis', fontsize=15, y=1.01)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'feature_boxplots.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_smoker_diagnosis() -> str:
    """Grouped bar chart — smoker status vs diagnosis."""
    ensure_dir()
    df = load_data()
    df_known = df[df['Diagnosis'] != 'UNKNOWN'].copy()

    ct = pd.crosstab(df_known['Smoker'], df_known['Diagnosis'])

    fig, ax = plt.subplots(figsize=(7, 5))
    ct.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax.set_xlabel('Smoker Status')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Smoker Status vs Diagnosis', fontsize=14)
    ax.set_xticklabels(ct.index, rotation=0)
    ax.legend(title='Diagnosis')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'smoker_diagnosis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_age_bmi_scatter() -> str:
    """Scatter plot of Age vs BMI coloured by Diagnosis."""
    ensure_dir()
    df = load_data()
    df_known = df[df['Diagnosis'] != 'UNKNOWN'].copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {'DIABETES': '#3498db', 'HEART DISEASE': '#e74c3c'}
    for diag, grp in df_known.groupby('Diagnosis'):
        ax.scatter(grp['Age'], grp['BMI'], label=diag, alpha=0.7,
                   color=palette[diag], edgecolors='black', linewidths=0.4)

    ax.set_xlabel('Age')
    ax.set_ylabel('BMI')
    ax.set_title('Age vs BMI by Diagnosis', fontsize=14)
    ax.legend(title='Diagnosis')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'age_bmi_scatter.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_gender_diagnosis() -> str:
    """Grouped bar chart — gender vs diagnosis."""
    ensure_dir()
    df = load_data()
    df_known = df[df['Diagnosis'] != 'UNKNOWN'].copy()

    ct = pd.crosstab(df_known['Gender'], df_known['Diagnosis'])

    fig, ax = plt.subplots(figsize=(7, 5))
    ct.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Gender vs Diagnosis', fontsize=14)
    ax.set_xticklabels(ct.index, rotation=0)
    ax.legend(title='Diagnosis')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(EXPLORATION_DIR, 'gender_diagnosis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_all():
    """Generate all exploration charts and return their paths."""
    return {
        'correlation_heatmap': generate_correlation_heatmap(),
        'diagnosis_distribution': generate_diagnosis_distribution(),
        'feature_boxplots': generate_feature_boxplots(),
        'smoker_diagnosis': generate_smoker_diagnosis(),
        'age_bmi_scatter': generate_age_bmi_scatter(),
        'gender_diagnosis': generate_gender_diagnosis(),
    }


if __name__ == '__main__':
    paths = generate_all()
    for name, path in paths.items():
        print(f'{name}: {path}')
