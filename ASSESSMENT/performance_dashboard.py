"""
Performance dashboard for the Random Forest classifier.

Reads pre-computed metrics from model_outputs/performance.json (written by
predictor.train()) and produces a set of Matplotlib charts:

    perf_roc_curve.png          — ROC curve with AUC annotation
    perf_train_test_acc.png     — Train vs test accuracy (overfitting indicator)
    perf_per_class_metrics.png  — Precision / Recall / F1 per class
    perf_confidence_hist.png    — Histogram of prediction confidence scores
    perf_cv_scores.png          — 5-fold CV accuracy + ROC-AUC bars
    perf_model_comparison.png   — All candidate models compared by CV acc + ROC-AUC
    perf_loo_cv.png             — Leave-One-Out CV accuracy vs 5-fold CV vs hold-out
    perf_summary_dashboard.png  — All five panels in a single figure
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from predictor import load_performance, load_comparison, MODEL_DIR

PERF_DIR = os.path.join(MODEL_DIR, 'performance_charts')
OVERFITTING_WARN_THRESHOLD = 20.0   # gap (%) above which we flag overfitting


def _ensure_dir():
    os.makedirs(PERF_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Individual charts
# ──────────────────────────────────────────────────────────────────────────────

def chart_roc_curve(perf: dict) -> str:
    """ROC curve with AUC annotation and random-baseline diagonal."""
    _ensure_dir()
    fpr = perf['fpr']
    tpr = perf['tpr']
    auc = perf['roc_auc']
    cv_auc = np.mean(perf['cv_roc'])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2980b9', lw=2,
            label=f'Hold-out ROC (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2980b9')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve\n(CV mean AUC = {cv_auc:.2f})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    out = os.path.join(PERF_DIR, 'perf_roc_curve.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_train_test_accuracy(perf: dict) -> str:
    """
    Side-by-side bar comparing train accuracy vs test accuracy.
    Highlights overfitting if the gap exceeds OVERFITTING_WARN_THRESHOLD.
    """
    _ensure_dir()
    train_acc = perf['train_acc'] * 100
    test_acc  = perf['test_acc']  * 100
    gap       = perf['overfitting_gap']
    overfit   = gap > OVERFITTING_WARN_THRESHOLD

    colours = ['#27ae60', '#e74c3c' if overfit else '#2980b9']
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Train Accuracy', 'Test Accuracy'],
                  [train_acc, test_acc], color=colours, edgecolor='black', width=0.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=10)
    ax.set_ylim([0, 115])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Train vs Test Accuracy\n(Overfitting Indicator)')
    ax.axhline(50, color='grey', lw=1, linestyle='--', label='Random baseline (50%)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)

    label = (f'Gap: {gap:.1f}% — {"OVERFITTING DETECTED" if overfit else "Acceptable"}'
             if gap > 0 else f'Gap: {gap:.1f}%')
    colour = '#c0392b' if overfit else '#27ae60'
    ax.text(0.5, 0.96, label, ha='center', va='top',
            transform=ax.transAxes, fontsize=9,
            color=colour, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdf2f2' if overfit else '#eafaf1',
                      edgecolor=colour, alpha=0.9))
    plt.tight_layout()
    out = os.path.join(PERF_DIR, 'perf_train_test_acc.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_per_class_metrics(perf: dict) -> str:
    """Grouped bar: Precision / Recall / F1 per class."""
    _ensure_dir()
    classes = perf['class_names']
    metrics = ['precision', 'recall', 'f1']
    colours = ['#3498db', '#e67e22', '#27ae60']

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        vals = [perf['per_class'][cls][metric] for cls in classes]
        rects = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                       color=colour, edgecolor='black')
        ax.bar_label(rects, fmt='%.0f%%', padding=2, fontsize=8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score (%)')
    ax.set_title('Per-Class Precision / Recall / F1')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 115])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(50, color='grey', lw=1, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PERF_DIR, 'perf_per_class_metrics.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_confidence_histogram(perf: dict) -> str:
    """
    Histogram of prediction confidence (max class probability) on the test set.
    High confidence + low accuracy = miscalibrated / overfitting model.
    """
    _ensure_dir()
    scores = perf['confidence_scores']
    mean_c = perf['mean_confidence']

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(scores, bins=10, range=(50, 100), color='#9b59b6',
            edgecolor='black', alpha=0.8)
    ax.axvline(mean_c, color='#c0392b', lw=2, linestyle='--',
               label=f'Mean = {mean_c:.1f}%')
    ax.set_xlabel('Confidence (max class probability, %)')
    ax.set_ylabel('Number of Test Predictions')
    ax.set_title('Prediction Confidence Distribution\n(Test Set)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([50, 100])
    plt.tight_layout()
    out = os.path.join(PERF_DIR, 'perf_confidence_hist.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_cv_scores(perf: dict) -> str:
    """
    5-fold CV accuracy and ROC-AUC scores shown as individual bars
    with mean lines.  CV is the most reliable estimate on a small dataset.
    """
    _ensure_dir()
    cv_acc  = [s * 100 for s in perf['cv_acc']]
    cv_roc  = [s * 100 for s in perf['cv_roc']]
    folds   = [f'Fold {i+1}' for i in range(len(cv_acc))]

    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_acc = ax.bar(x - width/2, cv_acc, width, label='CV Accuracy',
                      color='#2980b9', edgecolor='black')
    bars_roc = ax.bar(x + width/2, cv_roc, width, label='CV ROC-AUC',
                      color='#e67e22', edgecolor='black')

    mean_acc = float(np.mean(cv_acc))
    mean_roc = float(np.mean(cv_roc))
    ax.axhline(mean_acc, color='#2980b9', lw=1.5, linestyle='--',
               label=f'Mean acc = {mean_acc:.1f}%')
    ax.axhline(mean_roc, color='#e67e22', lw=1.5, linestyle='--',
               label=f'Mean AUC = {mean_roc:.1f}%')
    ax.axhline(50, color='grey', lw=1, linestyle=':', alpha=0.6,
               label='Random baseline (50%)')

    ax.bar_label(bars_acc, fmt='%.0f%%', padding=2, fontsize=8)
    ax.bar_label(bars_roc, fmt='%.0f%%', padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel('Score (%)')
    ax.set_ylim([0, 115])
    ax.set_title('5-Fold Cross-Validation Scores\n(Best estimate of generalisation)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PERF_DIR, 'perf_cv_scores.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Summary dashboard (all panels in one figure)
# ──────────────────────────────────────────────────────────────────────────────

def chart_summary_dashboard(perf: dict) -> str:
    """Composite 2×3 dashboard combining all five performance charts."""
    _ensure_dir()
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle('Model Performance Dashboard — Random Forest Classifier',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.08)

    # ── Panel 1: Train vs Test accuracy ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    train_acc = perf['train_acc'] * 100
    test_acc  = perf['test_acc']  * 100
    gap       = perf['overfitting_gap']
    overfit   = gap > OVERFITTING_WARN_THRESHOLD
    colours   = ['#27ae60', '#e74c3c' if overfit else '#2980b9']
    bars = ax1.bar(['Train', 'Test'], [train_acc, test_acc],
                   color=colours, edgecolor='black', width=0.5)
    ax1.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    ax1.set_ylim([0, 120])
    ax1.axhline(50, color='grey', lw=1, linestyle='--', alpha=0.6)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Train vs Test Accuracy', fontsize=10)
    c = '#c0392b' if overfit else '#27ae60'
    ax1.text(0.5, 0.93, f'Gap: {gap:.1f}%', ha='center', va='top',
             transform=ax1.transAxes, fontsize=8, color=c, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # ── Panel 2: ROC curve ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(perf['fpr'], perf['tpr'], color='#2980b9', lw=2,
             label=f'AUC={perf["roc_auc"]:.2f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.fill_between(perf['fpr'], perf['tpr'], alpha=0.1, color='#2980b9')
    ax2.set_xlabel('FPR', fontsize=9)
    ax2.set_ylabel('TPR', fontsize=9)
    ax2.set_title('ROC Curve (hold-out)', fontsize=10)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(alpha=0.3)

    # ── Panel 3: Per-class metrics ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    classes = perf['class_names']
    x = np.arange(len(classes))
    w = 0.25
    for i, (metric, colour) in enumerate(
            zip(['precision', 'recall', 'f1'], ['#3498db', '#e67e22', '#27ae60'])):
        vals = [perf['per_class'][cls][metric] for cls in classes]
        ax3.bar(x + i * w, vals, w, label=metric.capitalize(),
                color=colour, edgecolor='black')
    ax3.set_xticks(x + w)
    ax3.set_xticklabels([c.replace(' ', '\n') for c in classes], fontsize=8)
    ax3.set_ylim([0, 110])
    ax3.set_ylabel('Score (%)', fontsize=9)
    ax3.set_title('Per-Class P / R / F1', fontsize=10)
    ax3.legend(fontsize=7)
    ax3.grid(axis='y', alpha=0.3)

    # ── Panel 4: Confidence histogram ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(perf['confidence_scores'], bins=10, range=(50, 100),
             color='#9b59b6', edgecolor='black', alpha=0.8)
    ax4.axvline(perf['mean_confidence'], color='#c0392b', lw=2, linestyle='--',
                label=f'Mean={perf["mean_confidence"]:.0f}%')
    ax4.set_xlabel('Confidence (%)', fontsize=9)
    ax4.set_ylabel('Count', fontsize=9)
    ax4.set_title('Prediction Confidence', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # ── Panel 5: CV accuracy bars ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    cv_acc = [s * 100 for s in perf['cv_acc']]
    folds = [f'F{i+1}' for i in range(len(cv_acc))]
    bar_cols = ['#27ae60' if s >= 50 else '#e74c3c' for s in cv_acc]
    ax5.bar(folds, cv_acc, color=bar_cols, edgecolor='black')
    ax5.axhline(np.mean(cv_acc), color='#2c3e50', lw=2, linestyle='--',
                label=f'Mean={np.mean(cv_acc):.1f}%')
    ax5.axhline(50, color='grey', lw=1, linestyle=':', alpha=0.6)
    ax5.set_ylim([0, 115])
    ax5.set_ylabel('Accuracy (%)', fontsize=9)
    ax5.set_title('5-Fold CV Accuracy', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(axis='y', alpha=0.3)

    # ── Panel 6: CV ROC-AUC bars ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    cv_roc = [s * 100 for s in perf['cv_roc']]
    bar_cols_roc = ['#27ae60' if s >= 50 else '#e74c3c' for s in cv_roc]
    ax6.bar(folds, cv_roc, color=bar_cols_roc, edgecolor='black')
    ax6.axhline(np.mean(cv_roc), color='#e67e22', lw=2, linestyle='--',
                label=f'Mean={np.mean(cv_roc):.1f}%')
    ax6.axhline(50, color='grey', lw=1, linestyle=':', alpha=0.6)
    ax6.set_ylim([0, 115])
    ax6.set_ylabel('ROC-AUC (%)', fontsize=9)
    ax6.set_title('5-Fold CV ROC-AUC', fontsize=10)
    ax6.legend(fontsize=8)
    ax6.grid(axis='y', alpha=0.3)

    out = os.path.join(PERF_DIR, 'perf_summary_dashboard.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_model_comparison(comp: dict) -> str:
    """
    Grouped horizontal bar chart comparing all candidate models by
    5-fold CV accuracy and CV ROC-AUC.  The best model is highlighted.
    """
    _ensure_dir()
    models     = comp['models']
    best_name  = comp['best_model']
    names      = list(models.keys())
    acc_means  = [models[n]['cv_acc_mean'] for n in names]
    roc_means  = [models[n]['cv_roc_mean'] for n in names]
    acc_stds   = [models[n]['cv_acc_std']  for n in names]
    roc_stds   = [models[n]['cv_roc_std']  for n in names]

    y      = np.arange(len(names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_acc = ax.barh(y - height / 2, acc_means, height,
                       xerr=acc_stds, capsize=4,
                       color='#2980b9', edgecolor='black', label='CV Accuracy')
    bars_roc = ax.barh(y + height / 2, roc_means, height,
                       xerr=roc_stds, capsize=4,
                       color='#e67e22', edgecolor='black', label='CV ROC-AUC')

    # Highlight best model rows
    best_idx = names.index(best_name)
    for bar in [bars_acc[best_idx], bars_roc[best_idx]]:
        bar.set_edgecolor('#c0392b')
        bar.set_linewidth(2)

    ax.axvline(50, color='grey', lw=1, linestyle='--', alpha=0.7, label='Random baseline (50%)')
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Score (%)')
    ax.set_title('Model Comparison — 5-Fold CV (± std)\nBest model highlighted with red border',
                 fontsize=12)
    ax.set_xlim([0, 110])
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Value labels
    for bar_group, stds in [(bars_acc, acc_stds), (bars_roc, roc_stds)]:
        for bar, std in zip(bar_group, stds):
            w = bar.get_width()
            ax.text(w + std + 1, bar.get_y() + bar.get_height() / 2,
                    f'{w:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    out = os.path.join(PERF_DIR, 'perf_model_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def chart_loo_cv(perf: dict) -> str:
    """
    Bar chart comparing three accuracy estimates:
      - Single 25% hold-out split
      - 5-fold CV mean
      - Leave-One-Out CV
    LOO uses all 52 samples and gives the lowest-variance estimate.
    """
    _ensure_dir()
    test_acc  = perf['test_acc']  * 100
    cv_mean   = float(np.mean(perf['cv_acc'])) * 100
    loo_acc   = perf['loo_acc']   * 100

    labels  = ['Hold-out\n(25% split)', '5-Fold CV\nmean', 'Leave-One-Out\nCV']
    values  = [test_acc, cv_mean, loo_acc]
    colours = ['#e74c3c', '#2980b9', '#27ae60']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor='black', width=0.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=10)
    ax.axhline(50, color='grey', lw=1, linestyle='--', label='Random baseline (50%)')
    ax.set_ylim([0, 115])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Estimate Comparison\n(Hold-out vs 5-Fold CV vs LOO)', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    note = ('LOO uses all data for training each fold\n'
            '→ most reliable estimate on small datasets')
    ax.text(0.98, 0.97, note, ha='right', va='top',
            transform=ax.transAxes, fontsize=8, color='#555',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ccc'))
    plt.tight_layout()
    out = os.path.join(PERF_DIR, 'perf_loo_cv.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def generate_all() -> dict[str, str]:
    """Generate all performance charts and return a name → path mapping."""
    perf = load_performance()
    comp = load_comparison()
    return {
        'roc_curve':          chart_roc_curve(perf),
        'train_test_acc':     chart_train_test_accuracy(perf),
        'per_class_metrics':  chart_per_class_metrics(perf),
        'confidence_hist':    chart_confidence_histogram(perf),
        'cv_scores':          chart_cv_scores(perf),
        'model_comparison':   chart_model_comparison(comp),
        'loo_cv':             chart_loo_cv(perf),
        'summary_dashboard':  chart_summary_dashboard(perf),
    }


if __name__ == '__main__':
    paths = generate_all()
    for name, path in paths.items():
        print(f'{name}: {path}')
