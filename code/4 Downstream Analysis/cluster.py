"""Clustering Analysis Module"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from load_data import patients


def load_latent_features(result_path, encoder_key):
    """Load latent features from saved file."""
    latent_z = np.load(result_path)
    return latent_z[encoder_key][1, :][patients, :]


def find_optimal_clusters(data, cluster_range):
    """Find optimal number of clusters using elbow method."""
    sse_values = []
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(data)
        dist = np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)
        sse_values.append(dist.sum() / data.shape[0])
    
    kn = KneeLocator(cluster_range, sse_values, curve='convex', 
                     direction='decreasing', S=1.0)
    optimal_k = kn.elbow if kn.elbow else 2
    
    return optimal_k, sse_values


def plot_elbow(cluster_range, sse_values, optimal_k, save_path="elbow.png"):
    """Plot elbow curve for cluster number selection."""
    plt.figure(figsize=(4, 5))
    
    point_colors = ['red' if c == optimal_k else 'black' for c in cluster_range]
    point_sizes = [120 if c == optimal_k else 60 for c in cluster_range]
    
    plt.plot(cluster_range, sse_values, 'k-', linewidth=3, zorder=1)
    plt.scatter(cluster_range, sse_values, marker='^', 
                c=point_colors, s=point_sizes, zorder=2)
    
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    
    plt.xlabel('K', fontsize=25, fontweight='bold')
    plt.ylabel("Average Distortion", fontsize=25, fontweight='bold')
    plt.xticks([2, 5, 10], fontsize=20, fontweight='bold')
    plt.yticks([1.5, 2.0, 2.5], fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


def plot_tsne_clusters(data, labels, n_clusters, save_path="Cluster.png"):
    """Plot t-SNE visualization of clusters."""
    plt.figure(figsize=(6, 6))
    
    color_palette = [
        (0.6, 0.6, 0.6),    # Gray
        (0.35, 0.6, 0.6),   # Teal
        (0.8, 0.4, 0.4),    # Red
        (0.4, 0.8, 0.4)     # Green
    ]
    
    tsne = TSNE(random_state=42)
    data_tsne = tsne.fit_transform(data)
    
    for i in range(n_clusters):
        cluster_data = data_tsne[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    color=color_palette[i % len(color_palette)],
                    marker='o', s=15, label=f'Subtype {i+1}')
    
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    
    plt.xlabel('t-SNE 1', fontsize=25, fontweight='bold')
    plt.ylabel("t-SNE 2", fontsize=25, fontweight='bold')
    plt.xticks([-40, 0, 40], fontsize=20, fontweight='bold')
    plt.yticks([-10, 0, 10], fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


def main():
    # Configuration
    result_path = '../d_results_of_z/latent_z_train.npz'
    encoder_key = 'Z_cvae_sencoder'
    cluster_range = range(1, 10)
    
    # Load data
    data = load_latent_features(result_path, encoder_key)
    
    # Find optimal cluster number
    optimal_k, sse_values = find_optimal_clusters(data, cluster_range)
    print(f"Optimal clusters detected: {optimal_k}")
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, n_init='auto', 
                    max_iter=300, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    # Print cluster statistics
    for i in range(optimal_k):
        print(f"Subtype {i+1} count: {(labels == i).sum()}")
    
    # Generate visualizations
    plot_elbow(cluster_range, sse_values, optimal_k)
    plot_tsne_clusters(data, labels, optimal_k)


if __name__ == "__main__":
    main()
