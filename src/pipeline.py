"""
Modul Pipeline Utama (CRISP-DM End-to-End Runner)
Menjalankan seluruh siklus data mining:
Data Ingestion -> Preprocessing -> Feature Engineering -> Clustering ->
Targeting Error Analysis -> Priority Scoring -> Classification -> Export.
"""

import os
import argparse
import pandas as pd
import numpy as np

from src.scrape_real_data import generate_bps_real_dataset
from src.preprocessing import BPSDataPreprocessor
from src.features import compute_bps_regional_features
from src.clustering import BPSClusterEngine
from src.evaluation import benchmark_clustering_algorithms, compute_clustering_metrics, evaluate_k_range
from src.targeting import (
    assign_ordered_cluster_labels,
    compute_cluster_profiles,
    analyze_targeting_errors,
    compute_welfare_priority_ranking
)
from src.classification import WelfareClassifierEngine

def run_pipeline(
    raw_data_path: str = "data/raw/real_bps_kabupaten_kota.csv",
    processed_data_path: str = "data/processed/hasil_audit_bansos_wilayah.csv",
    n_clusters: int = 4,
    random_state: int = 42
):
    print("=" * 70)
    print("      PIPELINE DATA MINING BANSOS BERBASIS DATA RIIL BPS")
    print("=" * 70)

    # 1. Data Ingestion
    if not os.path.exists(raw_data_path):
        print(f"[INFO] Dataset '{raw_data_path}' belum ditemukan. Menjalankan scraper/builder data BPS...")
        df_raw = generate_bps_real_dataset(output_path=raw_data_path, seed=random_state)
    else:
        print(f"[1/8] Memuat dataset riil BPS dari '{raw_data_path}'...")
        df_raw = pd.read_csv(raw_data_path)
    print(f"      Total data: {len(df_raw)} Kabupaten/Kota se-Indonesia.")

    # 2. Preprocessing
    print("[2/8] Menjalankan pembersihan data dan penanganan outlier...")
    preprocessor = BPSDataPreprocessor(scaler_type="standard")
    df_clean = preprocessor.clean_and_impute(df_raw)
    df_clean = preprocessor.handle_outliers_iqr(df_clean)

    # 3. Feature Engineering
    print("[3/8] Menghitung Indeks Kemiskinan Multidimensi (IKM) & Fitur Audit Bansos...")
    df_feat = compute_bps_regional_features(df_clean)

    # Fitur untuk clustering
    clustering_features = [
        "persentase_penduduk_miskin",
        "garis_kemiskinan",
        "indeks_kedalaman_p1",
        "indeks_keparahan_p2",
        "pengeluaran_per_kapita",
        "indeks_pembangunan_manusia_ipm",
        "persentase_sanitasi_layak",
        "persentase_air_minum_layak",
        "persentase_rumah_layak_huni",
        "indeks_infrastruktur_dasar",
        "indeks_kemiskinan_multidimensi_ikm"
    ]
    
    X_scaled, df_scaled = preprocessor.scale_features(df_feat, clustering_features)

    # 4. Clustering & Evaluation Benchmarking
    print(f"[4/8] Melakukan evaluasi metrik dan klasterisasi K-Means (k={n_clusters})...")
    benchmark_df = benchmark_clustering_algorithms(X_scaled, n_clusters=n_clusters, random_state=random_state)
    print("\n[HASIL BENCHMARK ALGORITMA]")
    print(benchmark_df.to_string(index=False))

    cluster_engine = BPSClusterEngine(n_clusters=n_clusters, random_state=random_state)
    raw_labels = cluster_engine.fit_kmeans(X_scaled)
    df_feat["cluster_raw"] = raw_labels

    metrics = compute_clustering_metrics(X_scaled, raw_labels)
    print(f"\n[METRIK K-MEANS TERBAIK]")
    print(f"- Silhouette Score      : {metrics['silhouette_score']} (Target: >= 0.40)")
    print(f"- Davies-Bouldin Index  : {metrics['davies_bouldin_index']} (Target: < 1.20)")
    print(f"- Calinski-Harabasz     : {metrics['calinski_harabasz_index']}")

    # PCA 2D & 3D
    df_pca_2d, df_pca_3d = cluster_engine.compute_pca_projections(X_scaled)
    df_feat["PCA1"] = df_pca_2d["PCA1"]
    df_feat["PCA2"] = df_pca_2d["PCA2"]
    df_feat["PCA3"] = df_pca_3d["PCA3"]

    os.makedirs("models", exist_ok=True)
    cluster_engine.save_model("models/kmeans_bps_model.joblib")

    # 5. Cluster Profiling & Auto-Labeling
    print("\n[5/8] Menyusun profil karakteristik klaster dan pelabelan otomatis...")
    df_labeled, label_mapping = assign_ordered_cluster_labels(
        df_feat,
        cluster_col="cluster_raw",
        ranking_metric="indeks_kemiskinan_multidimensi_ikm"
    )

    # 6. Targeting Error Analysis (Inclusion & Exclusion Error)
    print("[6/8] Menganalisis ketepatan sasaran bansos (Inclusion & Exclusion Error)...")
    error_analysis = analyze_targeting_errors(df_labeled)
    print(f"- Exclusion Error Wilayah: {error_analysis['exclusion_error_wilayah_pct']}% (Daerah Miskin yang defisit bansos)")
    print(f"- Inclusion Error Wilayah: {error_analysis['inclusion_error_wilayah_pct']}% (Daerah Sejahtera yang surplus bansos)")

    # 7. Welfare Priority Ranking (WPI)
    print("[7/8] Menghitung Welfare Priority Index (WPI) dan ranking rekomendasi alokasi...")
    df_final = compute_welfare_priority_ranking(df_labeled)

    # 8. Supervised Classification
    print("[8/8] Melatih model klasifikasi prioritas bansos (Random Forest & XGBoost)...")
    classifier = WelfareClassifierEngine(random_state=random_state)
    X_train, X_test, y_train, y_test = classifier.prepare_data(df_final, feature_cols=clustering_features)
    classifier.train_models(X_train, y_train)
    cls_results = classifier.evaluate_models(X_test, y_test)
    classifier.save_models()

    print(f"- F1-Score Random Forest: {cls_results['Random Forest']['f1_score']}")
    print(f"- F1-Score XGBoost      : {cls_results['XGBoost']['f1_score']}")

    # Export Hasil
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_final.to_csv(processed_data_path, index=False)
    print(f"\n[SUKSES] Seluruh pipeline selesai! Hasil tersimpan di '{processed_data_path}'.")
    print("=" * 70)
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Data Mining Runner Bansos BPS")
    parser.add_argument("--raw", type=str, default="data/raw/real_bps_kabupaten_kota.csv")
    parser.add_argument("--output", type=str, default="data/processed/hasil_audit_bansos_wilayah.csv")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_pipeline(
        raw_data_path=args.raw,
        processed_data_path=args.output,
        n_clusters=args.k,
        random_state=args.seed
    )
