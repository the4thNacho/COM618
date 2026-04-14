"""
Interactive K-Means Clustering for Medical Data Exploration.

Provides dynamic clustering analysis with real-time parameter adjustment
for discovering patterns in cleaned medical datasets.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

CLEANED_PATH = os.path.join(os.path.dirname(__file__), 'realworld_medical_dirty_cleaned_1.csv')
EXPLORATION_VISUALS = os.path.join(os.path.dirname(__file__), 'exploration_visuals')


def load_and_prepare_data():
    """Load medical data and prepare for clustering analysis."""
    
    # Ensure output directory exists
    os.makedirs(EXPLORATION_VISUALS, exist_ok=True)
    
    # Load cleaned dataset
    df = pd.read_csv(CLEANED_PATH, keep_default_na=False, na_values=[''])
    
    # Focus on labeled samples for pattern discovery
    labeled_df = df[df['Diagnosis'] != 'UNKNOWN'].copy()
    
    # Prepare numerical features for clustering
    numerical_cols = ['Age', 'Blood_Pressure', 'Cholesterol', 'BMI']
    X = labeled_df[numerical_cols].values
    
    # Handle any remaining missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Feature scaling (essential for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create binary encoding for diagnosis
    y = (labeled_df['Diagnosis'] == 'HEART DISEASE').astype(int)
    
    return X_scaled, y, labeled_df, numerical_cols, scaler


def perform_kmeans_clustering(n_clusters=3, max_iter=300, random_state=42):
    """
    Perform K-means clustering with specified parameters.
    
    Returns clustering results, metrics, and visualization paths.
    """
    X_scaled, y_true, labeled_df, feature_cols, scaler = load_and_prepare_data()
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    inertia = kmeans.inertia_
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Generate visualizations
    viz_paths = create_clustering_visualizations(
        X_scaled, X_pca, cluster_labels, y_true, labeled_df, 
        kmeans, pca, n_clusters, feature_cols
    )
    
    # Cluster statistics
    cluster_stats = analyze_clusters(labeled_df, cluster_labels, feature_cols)
    
    results = {
        'n_clusters': n_clusters,
        'max_iter': max_iter,
        'silhouette_score': round(silhouette_avg, 3),
        'davies_bouldin_score': round(davies_bouldin, 3),
        'inertia': round(inertia, 2),
        'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(n_clusters)],
        'cluster_stats': cluster_stats,
        'visualizations': viz_paths,
        'pca_explained_variance': pca.explained_variance_ratio_.tolist()
    }
    
    return results


def create_clustering_visualizations(X_scaled, X_pca, cluster_labels, y_true, labeled_df, kmeans, pca, n_clusters, feature_cols):
    """Create comprehensive clustering visualizations."""
    
    viz_paths = {}
    
    # 1. PCA Scatter Plot with Clusters
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'K-Means Clusters (k={n_clusters})')
    plt.grid(True, alpha=0.3)
    
    # Add cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.legend()
    
    # Subplot 2: True Diagnosis
    plt.subplot(1, 2, 2)
    diagnosis_colors = ['skyblue', 'lightcoral']
    diagnosis_labels = ['Diabetes', 'Heart Disease']
    scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[diagnosis_colors[i] for i in y_true], alpha=0.7, s=60)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('True Diagnoses')
    plt.grid(True, alpha=0.3)
    
    # Create legend for true diagnoses
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=diagnosis_colors[i], 
                             markersize=8, label=diagnosis_labels[i]) for i in range(2)]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    viz_paths['pca_clusters'] = os.path.join(EXPLORATION_VISUALS, 'kmeans_pca_clusters.png')
    plt.savefig(viz_paths['pca_clusters'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Distribution by Cluster
    plt.figure(figsize=(14, 8))
    
    for i, feature in enumerate(feature_cols):
        plt.subplot(2, 2, i + 1)
        
        # Create DataFrame for easy plotting
        plot_df = pd.DataFrame({
            'Feature': labeled_df[feature].values,
            'Cluster': cluster_labels
        })
        
        # Box plot by cluster
        sns.boxplot(data=plot_df, x='Cluster', y='Feature', palette='viridis')
        plt.title(f'{feature} by Cluster')
        plt.ylabel(feature)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature Distributions Across {n_clusters} Clusters', fontsize=16, y=1.02)
    plt.tight_layout()
    viz_paths['feature_distributions'] = os.path.join(EXPLORATION_VISUALS, 'kmeans_feature_distributions.png')
    plt.savefig(viz_paths['feature_distributions'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Cluster Composition (Diagnosis Distribution)
    plt.figure(figsize=(10, 6))
    
    # Prepare data for stacked bar chart
    cluster_diagnosis = pd.crosstab(cluster_labels, labeled_df['Diagnosis'])
    
    ax = cluster_diagnosis.plot(kind='bar', stacked=True, color=['skyblue', 'lightcoral'], figsize=(10, 6))
    plt.title(f'Diagnosis Composition by Cluster (k={n_clusters})')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Patients')
    plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for container in ax.containers:
        ax.bar_label(container, label_type='center')
    
    plt.tight_layout()
    viz_paths['cluster_composition'] = os.path.join(EXPLORATION_VISUALS, 'kmeans_cluster_composition.png')
    plt.savefig(viz_paths['cluster_composition'], dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_paths


def analyze_clusters(labeled_df, cluster_labels, feature_cols):
    """Analyze characteristics of each cluster."""
    
    cluster_stats = {}
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = labeled_df[cluster_mask]
        
        # Basic statistics
        stats = {
            'size': int(np.sum(cluster_mask)),
            'diagnosis_counts': cluster_data['Diagnosis'].value_counts().to_dict(),
            'feature_means': {}
        }
        
        # Feature means for this cluster
        for feature in feature_cols:
            stats['feature_means'][feature] = round(float(cluster_data[feature].mean()), 2)
        
        cluster_stats[f'cluster_{cluster_id}'] = stats
    
    return cluster_stats


def generate_elbow_plot(max_k=8):
    """Generate elbow plot to help determine optimal number of clusters."""
    
    X_scaled, _, _, _, _ = load_and_prepare_data()
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    
    # Create elbow plot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Inertia (WCSS)
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    elbow_path = os.path.join(EXPLORATION_VISUALS, 'kmeans_elbow_plot.png')
    plt.savefig(elbow_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Find suggested optimal k (highest silhouette score)
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[optimal_k_idx]
    
    return {
        'elbow_plot_path': 'kmeans_elbow_plot.png',
        'suggested_k': optimal_k,
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': [round(score, 3) for score in silhouette_scores]
    }


if __name__ == "__main__":
    # Generate elbow plot
    elbow_results = generate_elbow_plot()
    print("Elbow analysis completed:")
    print(f"Suggested optimal k: {elbow_results['suggested_k']}")
    
    # Run clustering with suggested k
    results = perform_kmeans_clustering(n_clusters=elbow_results['suggested_k'])
    print(f"\nClustering completed with k={results['n_clusters']}:")
    print(f"Silhouette Score: {results['silhouette_score']}")
    print(f"Davies-Bouldin Score: {results['davies_bouldin_score']}")
    print(f"Cluster sizes: {results['cluster_sizes']}")