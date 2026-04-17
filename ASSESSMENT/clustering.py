"""
K-Means clustering analysis for the Diabetes 130-US Hospitals dataset.

Used on the exploration page to let users interactively explore patient
groupings by adjusting k and the x/y axis features.

Approach:
  - Scale the numeric clinical features with StandardScaler
  - Fit KMeans on a sample (≤10k rows) for speed; label full dataset
  - Return base64-encoded PNG images for AJAX responses
  - Elbow/silhouette chart computed once and cached
"""

import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from cleaning import load_and_clean

# Features used as inputs to the K-means algorithm (all numeric, clinically meaningful)
CLUSTER_CORE_FEATURES = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'total_prior_visits',
    'num_meds_changed', 'num_meds_used', 'a1c_result_enc',
    'glu_serum_enc', 'insulin_enc',
]

# Feature options exposed to the user for the scatter axes
CLUSTER_DISPLAY_FEATURES = [
    ('age_numeric',        'Age'),
    ('time_in_hospital',   'Time in Hospital (days)'),
    ('num_medications',    'Number of Medications'),
    ('num_lab_procedures', 'Lab Procedures'),
    ('number_inpatient',   'Prior Inpatient Visits'),
    ('number_diagnoses',   'Number of Diagnoses'),
    ('total_prior_visits', 'Total Prior Visits'),
    ('num_meds_used',      'Medications Used'),
]

# Simple in-process cache so repeated requests don't re-scale data
_cache: dict = {}


def _load_and_scale():
    """Return (X_scaled, df, scaler) — cached after first call."""
    if 'X_scaled' in _cache:
        return _cache['X_scaled'], _cache['df'], _cache['scaler']

    df = load_and_clean()
    X = df[CLUSTER_CORE_FEATURES].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _cache['X_scaled'] = X_scaled
    _cache['df'] = df
    _cache['scaler'] = scaler
    return X_scaled, df, scaler


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Elbow + silhouette chart (k=2..8)
# ─────────────────────────────────────────────────────────────────────────────

def generate_elbow_chart() -> str:
    """Return base64 PNG of elbow + silhouette chart for k=2..8."""
    if 'elbow_b64' in _cache:
        return _cache['elbow_b64']

    X_scaled, _, _ = _load_and_scale()

    # Sample for speed — 8k rows is plenty for elbow estimation
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_scaled), min(8000, len(X_scaled)), replace=False)
    Xs = X_scaled[idx]

    K_range = range(2, 9)
    inertias, sil_scores = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(Xs, labels, sample_size=3000, random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(K_range), inertias, 'bo-', lw=2, ms=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inertia (within-cluster SS)', fontsize=11)
    axes[0].set_title('Elbow Method — Optimal K', fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(list(K_range))

    best_k_idx = int(np.argmax(sil_scores))
    best_k = list(K_range)[best_k_idx]
    axes[1].plot(list(K_range), sil_scores, 'ro-', lw=2, ms=8)
    axes[1].axvline(best_k, color='red', linestyle='--', alpha=0.5,
                    label=f'Best k={best_k}')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Score vs K', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(list(K_range))

    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _cache['elbow_b64'] = b64
    return b64


# ─────────────────────────────────────────────────────────────────────────────
# Cluster scatter (k variable, user-chosen axes)
# ─────────────────────────────────────────────────────────────────────────────

def generate_cluster_scatter(k: int, x_feature: str, y_feature: str) -> tuple:
    """
    Fit KMeans(k) and return (scatter_b64, pca_b64, metrics_dict).

    metrics_dict keys: k, silhouette, cluster_stats (list of dicts)
    """
    k = max(2, min(8, k))

    # Validate feature names
    valid = {f for f, _ in CLUSTER_DISPLAY_FEATURES}
    if x_feature not in valid:
        x_feature = 'age_numeric'
    if y_feature not in valid:
        y_feature = 'time_in_hospital'

    X_scaled, df, _ = _load_and_scale()

    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)

    sil = float(silhouette_score(X_scaled, labels, sample_size=5000, random_state=42))

    x_vals = df[x_feature].values
    y_vals = df[y_feature].values
    readmit = df['readmitted_early'].values

    # Per-cluster statistics
    cluster_stats = []
    for c in range(k):
        mask = labels == c
        n = int(mask.sum())
        rate = float(readmit[mask].mean() * 100)
        cluster_stats.append({
            'cluster': c,
            'n': n,
            'readmission_rate': round(rate, 1),
            'pct_of_total': round(n / len(labels) * 100, 1),
        })

    cluster_stats.sort(key=lambda s: s['readmission_rate'], reverse=True)

    # ── Scatter plot ───────────────────────────────────────────────────────────
    colours = plt.cm.tab10(np.linspace(0, 1, k))
    x_label = dict(CLUSTER_DISPLAY_FEATURES).get(x_feature, x_feature)
    y_label = dict(CLUSTER_DISPLAY_FEATURES).get(y_feature, y_feature)

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        mask = labels == c
        stat = next(s for s in cluster_stats if s['cluster'] == c)
        ax.scatter(x_vals[mask], y_vals[mask],
                   color=colours[c], alpha=0.3, s=15,
                   label=f'Cluster {c}  n={stat["n"]:,}  {stat["readmission_rate"]}% readmit')

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'K-Means Clustering (k={k})', fontsize=13)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    scatter_b64 = _fig_to_b64(fig)

    # ── PCA 2-D projection ─────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for c in range(k):
        mask = labels == c
        stat = next(s for s in cluster_stats if s['cluster'] == c)
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=colours[c], alpha=0.3, s=15,
                    label=f'Cluster {c}  {stat["readmission_rate"]}% readmit')

    ax2.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)', fontsize=11)
    ax2.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)', fontsize=11)
    ax2.set_title(f'PCA Projection — K-Means (k={k})', fontsize=13)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    pca_b64 = _fig_to_b64(fig2)

    metrics = {
        'k': k,
        'silhouette': round(sil, 3),
        'cluster_stats': cluster_stats,
        'pca_variance': round(float(sum(ev)) * 100, 1),
    }
    return scatter_b64, pca_b64, metrics
