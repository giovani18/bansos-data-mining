"""
Modul Klasterisasi (Clustering Engine)
Mengimplementasikan K-Means, Gaussian Mixture Models (GMM),
Agglomerative Hierarchical Clustering, serta reduksi dimensi PCA 2D/3D.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import joblib

class BPSClusterEngine:
    """
    Mesin klasterisasi tingkat kesejahteraan/kemiskinan daerah berbasis indikator BPS.
    """
    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.pca_2d = None
        self.pca_3d = None
        self.cluster_labels = None
        self.feature_names = None

    def fit_kmeans(self, X: np.ndarray) -> np.ndarray:
        """
        Melatih model K-Means Clustering.
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=self.random_state
        )
        self.cluster_labels = self.model.fit_predict(X)
        return self.cluster_labels

    def fit_gmm(self, X: np.ndarray) -> np.ndarray:
        """
        Melatih model Gaussian Mixture Models (GMM).
        """
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            max_iter=200,
            random_state=self.random_state
        )
        self.cluster_labels = self.model.fit_predict(X)
        return self.cluster_labels

    def fit_agglomerative(self, X: np.ndarray, linkage: str = "ward") -> np.ndarray:
        """
        Melatih model Agglomerative Hierarchical Clustering.
        """
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=linkage
        )
        self.cluster_labels = self.model.fit_predict(X)
        return self.cluster_labels

    def compute_pca_projections(self, X: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Menghitung proyeksi PCA 2D dan 3D untuk visualisasi sebaran klaster.
        """
        # PCA 2D
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        pca_2d_coords = self.pca_2d.fit_transform(X)
        df_pca_2d = pd.DataFrame(
            pca_2d_coords,
            columns=["PCA1", "PCA2"]
        )
        df_pca_2d["var_explained_2d"] = self.pca_2d.explained_variance_ratio_.sum()

        # PCA 3D
        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        pca_3d_coords = self.pca_3d.fit_transform(X)
        df_pca_3d = pd.DataFrame(
            pca_3d_coords,
            columns=["PCA1", "PCA2", "PCA3"]
        )
        df_pca_3d["var_explained_3d"] = self.pca_3d.explained_variance_ratio_.sum()

        return df_pca_2d, df_pca_3d

    def save_model(self, filepath: str = "models/kmeans_bps_model.joblib"):
        """
        Menyimpan model klasterisasi terlatih ke disk.
        """
        joblib.dump(self.model, filepath)
        print(f"[SUCCESS] Model tersimpan di '{filepath}'.")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Memuat model klasterisasi dari disk.
        """
        model = joblib.load(filepath)
        engine = cls(n_clusters=getattr(model, "n_clusters", 4))
        engine.model = model
        return engine
