"""
clustering.py
Modul sederhana untuk clustering dan evaluasi unsupervised learning
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


class ClusterAnalyzer:
    """Kelas untuk membantu proses clustering"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.model = None

    def prepare_features(self, df, columns=None, fit=True):
        """
        Menyiapkan fitur numerik untuk clustering.

        Jika `columns` tidak diisi, semua kolom numerik akan dipakai.
        """
        data = df.copy()
        if columns is None:
            columns = data.select_dtypes(include=["number"]).columns.tolist()

        X = data[columns].copy()
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=columns, index=data.index)

    def fit_kmeans(self, X, n_clusters=3, random_state=42):
        """Melatih model KMeans."""
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = self.model.fit_predict(X)
        return labels

    def evaluate_clustering(self, X, labels, y_true=None):
        """Menghitung silhouette score dan ARI bila label asli tersedia."""
        metrics = {
            "silhouette": silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan,
        }

        if y_true is not None:
            metrics["ari"] = adjusted_rand_score(y_true, labels)

        return metrics

    def find_best_k(self, X, k_range=range(2, 6), y_true=None):
        """Mencari jumlah cluster terbaik berdasarkan silhouette score."""
        results = []

        for k in k_range:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            score = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
            row = {"k": k, "silhouette": score}
            if y_true is not None:
                row["ari"] = adjusted_rand_score(y_true, labels)
            results.append(row)

        return pd.DataFrame(results).sort_values("silhouette", ascending=False)

    def reduce_to_2d(self, X):
        """Mengurangi fitur menjadi 2 dimensi untuk visualisasi."""
        return self.pca.fit_transform(X)
