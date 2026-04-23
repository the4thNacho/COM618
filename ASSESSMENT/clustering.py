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

# Features used as inputs to the K-means algorithm (all numeric, clinically meaningful).
# Excluded: number_outpatient (~98% zeros), number_emergency (~98% zeros),
# glu_serum_enc (~92% zeros) — near-zero variance even after scaling, adds noise.
CLUSTER_CORE_FEATURES = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures',
    'num_medications', 'number_inpatient', 'number_diagnoses',
    'total_prior_visits', 'num_meds_changed', 'num_meds_used',
    'a1c_result_enc', 'insulin_enc',
]

# Count features with heavy right-skew — log1p-transformed before scaling
_LOG_FEATURES = {
    'num_lab_procedures', 'num_medications', 'number_inpatient',
    'number_diagnoses', 'total_prior_visits', 'num_meds_used',
}

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
    X = df[CLUSTER_CORE_FEATURES].copy()
    # Fill remaining NaN with column median (missing ≠ 0 for count features)
    X = X.fillna(X.median())
    # Log-transform right-skewed count features to reduce outlier influence
    for col in _LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])
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
# Cluster results: PCA projection + cluster profile chart
# ─────────────────────────────────────────────────────────────────────────────

_PROFILE_LABELS = {
    'age_numeric':       'Age',
    'time_in_hospital':  'Time in Hospital',
    'num_lab_procedures':'Lab Procedures',
    'num_medications':   'Medications',
    'number_inpatient':  'Prior Inpatient',
    'number_diagnoses':  'Diagnoses',
    'total_prior_visits':'Prior Visits',
    'num_meds_changed':  'Meds Changed',
    'num_meds_used':     'Meds Used',
    'a1c_result_enc':    'A1C Result',
    'insulin_enc':       'Insulin',
}

CLUSTER_COLOURS = [
    '#e74c3c', '#2980b9', '#27ae60', '#8e44ad',
    '#e67e22', '#16a085', '#c0392b', '#2c3e50',
]


def generate_cluster_results(k: int) -> tuple:
    """
    Fit KMeans(k) and return (pca_b64, profile_b64, metrics_dict).

    pca_b64     — PCA 2-D projection coloured by cluster
    profile_b64 — horizontal grouped bar chart of normalised feature means
    metrics_dict keys: k, silhouette, cluster_stats
    """
    k = max(2, min(8, k))
    X_scaled, df, _ = _load_and_scale()

    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)

    sil = float(silhouette_score(X_scaled, labels, sample_size=5000, random_state=42))
    readmit = df['readmitted_early'].values

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

    colours = [CLUSTER_COLOURS[c % len(CLUSTER_COLOURS)] for c in range(k)]
    rank_label = {s['cluster']: i for i, s in enumerate(cluster_stats)}

    # ── PCA 2-D projection ─────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    # Subsample for rendering speed (up to 15k points)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_pca), min(15000, len(X_pca)), replace=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for c in range(k):
        mask_full = labels == c
        mask_sub = mask_full[idx]
        stat = next(s for s in cluster_stats if s['cluster'] == c)
        rank = rank_label[c] + 1
        ax.scatter(X_pca[idx][mask_sub, 0], X_pca[idx][mask_sub, 1],
                   color=colours[c], alpha=0.45, s=12, linewidths=0,
                   label=f'#{rank} Cluster {c}  ({stat["readmission_rate"]}% readmit, n={stat["n"]:,})')

    ax.set_xlabel(f'PC1 — {ev[0]*100:.1f}% variance explained', fontsize=11)
    ax.set_ylabel(f'PC2 — {ev[1]*100:.1f}% variance explained', fontsize=11)
    ax.set_title(f'PCA Projection of K-Means Clusters (k={k})', fontsize=13)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.95,
              title='Rank by readmission rate', title_fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    pca_b64 = _fig_to_b64(fig)

    # ── Cluster profile chart ──────────────────────────────────────────────────
    # Show normalised (z-score relative to overall mean) feature means per cluster
    feat_cols = list(CLUSTER_CORE_FEATURES)
    feat_labels = [_PROFILE_LABELS.get(f, f) for f in feat_cols]
    df_sub = df[feat_cols].copy()
    overall_mean = df_sub.mean()
    overall_std  = df_sub.std().replace(0, 1)

    n_feats = len(feat_cols)
    bar_h = 0.8 / k
    y_pos = np.arange(n_feats)

    fig2, ax2 = plt.subplots(figsize=(10, max(5, n_feats * 0.55 + 1)))
    for i, c in enumerate(range(k)):
        mask = labels == c
        means = df_sub[mask].mean()
        z = (means - overall_mean) / overall_std
        offset = (i - k / 2 + 0.5) * bar_h
        stat = next(s for s in cluster_stats if s['cluster'] == c)
        rank = rank_label[c] + 1
        ax2.barh(y_pos + offset, z.values, height=bar_h * 0.9,
                 color=colours[c], alpha=0.85,
                 label=f'#{rank} Cluster {c} ({stat["readmission_rate"]}% readmit)')

    ax2.axvline(0, color='#333', lw=1.2, ls='--', alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feat_labels, fontsize=10)
    ax2.set_xlabel('Z-score relative to overall mean\n(positive = above average, negative = below average)', fontsize=10)
    ax2.set_title(f'Cluster Feature Profiles — what defines each group (k={k})', fontsize=12)
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.95,
               title='Rank by readmission rate', title_fontsize=8)
    ax2.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    profile_b64 = _fig_to_b64(fig2)

    metrics = {
        'k': k,
        'silhouette': round(sil, 3),
        'cluster_stats': cluster_stats,
        'pca_variance': round(float(sum(ev)) * 100, 1),
    }
    return pca_b64, profile_b64, metrics
