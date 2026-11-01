"""
Modul Evaluasi Klaster (Clustering Evaluation & Benchmarking)
Menghitung Elbow Method (Inertia), Silhouette Score, Davies-Bouldin Index,
Calinski-Harabasz Index, serta tabel komparasi multi-algoritma.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_k_range(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Mengevaluasi performa K-Means untuk rentang k = k_min s.d. k_max
    menggunakan metrik Inertia (Elbow) dan Silhouette Score.
    """
    results = []
    
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=random_state)
        labels = km.fit_predict(X)
        
        inertia = km.inertia_
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        
        results.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil, 4),
            "davies_bouldin_index": round(db, 4),
            "calinski_harabasz_index": round(ch, 2)
        })
        
    return pd.DataFrame(results)


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Menghitung skor metrik kualitas klaster untuk label tertentu.
    """
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    
    return {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_index": round(ch, 2)
    }


def benchmark_clustering_algorithms(
    X: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Membandingkan performa berbagai algoritma clustering (K-Means, GMM, Agglomerative Ward/Complete)
    pada dataset yang sama dengan metrik Silhouette, Davies-Bouldin, dan Calinski-Harabasz.
    """
    models = {
        "K-Means": KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=random_state),
        "Gaussian Mixture Model (GMM)": GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state),
        "Agglomerative (Ward Linkage)": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
        "Agglomerative (Complete Linkage)": AgglomerativeClustering(n_clusters=n_clusters, linkage="complete"),
        "Agglomerative (Average Linkage)": AgglomerativeClustering(n_clusters=n_clusters, linkage="average"),
    }
    
    benchmark_list = []
    
    for name, model in models.items():
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X)
        else:
            labels = model.fit(X).predict(X)
            
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        
        # Evaluasi target PRD
        sil_pass = "Memenuhi (≥ 0.40)" if sil >= 0.40 else "Di bawah target"
        db_pass = "Memenuhi (< 1.20)" if db < 1.20 else "Di atas target"
        
        benchmark_list.append({
            "Algoritma": name,
            "Silhouette Score": round(sil, 4),
            "Status Silhouette": sil_pass,
            "Davies-Bouldin Index": round(db, 4),
            "Status Davies-Bouldin": db_pass,
            "Calinski-Harabasz Index": round(ch, 2)
        })
        
    df_benchmark = pd.DataFrame(benchmark_list)
    df_benchmark = df_benchmark.sort_values(by="Silhouette Score", ascending=False).reset_index(drop=True)
    return df_benchmark
