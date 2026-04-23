"""
Performance dashboard charts for the diabetes readmission Gradient Boosting model.

Charts written to model_outputs/performance_charts/:
    perf_roc_curve.png          — ROC curve with AUC
    perf_cv_scores.png          — Cross-validation accuracy and ROC-AUC
    perf_per_class.png          — Precision / Recall / F1 per class
    perf_feature_importance.png — Feature importance bar chart
    perf_train_test.png         — Train vs test accuracy (overfitting indicator)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from predictor import load_performance, MODEL_DIR

PERF_DIR = os.path.join(MODEL_DIR, 'performance_charts')


def _ensure_dir():
    os.makedirs(PERF_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────

def chart_cv_scores(perf: dict) -> str:
    _ensure_dir()
    cv_roc = np.array(perf['cv_roc'])
    cv_acc = np.array(perf['cv_acc'])
    folds = list(range(1, len(cv_roc) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(folds, cv_acc * 100, 'o-', color='#2980b9', label='Accuracy', lw=2)
    ax.plot(folds, cv_roc * 100, 's-', color='#e74c3c', label='ROC-AUC', lw=2)
    ax.axhline(cv_acc.mean() * 100, color='#2980b9', linestyle='--', alpha=0.5)
    ax.axhline(cv_roc.mean() * 100, color='#e74c3c', linestyle='--', alpha=0.5)
    ax.set_xlabel('CV Fold')
    ax.set_ylabel('Score (%)')
    ax.set_title('5-Fold Cross-Validation Scores', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    out = os.path.join(PERF_DIR, 'perf_cv_scores.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_per_class(perf: dict) -> str:
    _ensure_dir()
    per_class = perf.get('per_class', {})
    if not per_class:
        return ''

    classes    = list(per_class.keys())
    precision  = [per_class[c]['precision'] * 100 for c in classes]
    recall     = [per_class[c]['recall']    * 100 for c in classes]
    f1         = [per_class[c]['f1']        * 100 for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w,   precision, w, label='Precision', color='#2980b9', edgecolor='black')
    ax.bar(x,       recall,    w, label='Recall',    color='#27ae60', edgecolor='black')
    ax.bar(x + w,   f1,        w, label='F1-Score',  color='#e67e22', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Score (%)')
    ax.set_title('Per-Class Precision / Recall / F1', fontsize=13)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PERF_DIR, 'perf_per_class.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_train_test(perf: dict) -> str:
    _ensure_dir()
    train_acc = perf.get('train_acc', 0) * 100
    test_acc  = perf.get('test_acc',  0) * 100
    cv_acc    = np.mean(perf.get('cv_acc', [0])) * 100

    labels = ['Training', 'CV (mean)', 'Test (hold-out)']
    values = [train_acc, cv_acc, test_acc]
    colours = ['#2980b9', '#e67e22', '#27ae60']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor='black', width=0.45)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=11)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Train vs CV vs Test Accuracy', fontsize=13)
    ax.set_ylim(0, 100)
    ax.axhline(50, color='grey', linestyle='--', alpha=0.5, label='Random baseline (50%)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PERF_DIR, 'perf_train_test.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_roc_summary(perf: dict) -> str:
    """Bar chart comparing ROC-AUC: test, CV mean, and random baseline."""
    _ensure_dir()
    roc_auc = perf.get('roc_auc', 0) * 100
    cv_roc  = np.mean(perf.get('cv_roc', [0])) * 100

    labels = ['Random\nBaseline', 'CV ROC-AUC\n(mean)', 'Test\nROC-AUC']
    values = [50.0, cv_roc, roc_auc]
    colours = ['#bdc3c7', '#e67e22', '#2980b9']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor='black', width=0.45)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=11)
    ax.set_ylabel('ROC-AUC (%)')
    ax.set_title('ROC-AUC vs Random Baseline', fontsize=13)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PERF_DIR, 'perf_roc_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_all() -> dict:
    perf = load_performance()
    _ensure_dir()
    return {
        'cv_scores':    chart_cv_scores(perf),
        'per_class':    chart_per_class(perf),
        'train_test':   chart_train_test(perf),
        'roc_summary':  chart_roc_summary(perf),
    }


if __name__ == '__main__':
    paths = generate_all()
    for name, path in paths.items():
        print(f'{name}: {path}')
