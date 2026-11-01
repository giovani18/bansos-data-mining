"""
Skrip Pembangkit Jupyter Notebooks (CRISP-DM Workflow)
Menghasilkan 4 Jupyter Notebooks terstruktur:
- 01_eda.ipynb
- 02_preprocessing.ipynb
- 03_clustering.ipynb
- 04_targeting_analysis.ipynb
"""

import json
import os

def create_notebook(cells: list, output_path: str):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Notebook berhasil dibuat di '{output_path}'.")


def build_all_notebooks():
    # --- 1. 01_eda.ipynb ---
    cells_eda = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01. Exploratory Data Analysis (EDA) - Data Riil BPS\n",
                "## Analisis Efektivitas dan Ketepatan Sasaran Bantuan Sosial\n",
                "\n",
                "Notebook ini memuat tahapan **Data Understanding** sesuai metodologi CRISP-DM:\n",
                "1. Memuat dataset 514 Kabupaten/Kota se-Indonesia hasil scraping BPS.\n",
                "2. Menelaah struktur data, tipe variabel, dan kelengkapan informasi.\n",
                "3. Visualisasi distribusi indikator kemiskinan ($P_0, P_1, P_2$), pengeluaran, IPM, dan fasilitas perumahan.\n",
                "4. Analisis korelasi antar-indikator kemiskinan dan alokasi bantuan sosial."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import plotly.express as px\n",
                "\n",
                "# Set plot styling\n",
                "sns.set_theme(style='whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "\n",
                "# 1. Memuat Data\n",
                "df = pd.read_csv('../data/raw/real_bps_kabupaten_kota.csv')\n",
                "print(f'Dimensi Dataset: {df.shape[0]} Baris, {df.shape[1]} Kolom')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Ringkasan Statistik Deskriptif\n",
                "print('=== RINGKASAN STATISTIK INDIKATOR UTAMA BPS ===')\n",
                "cols_key = ['persentase_penduduk_miskin', 'garis_kemiskinan', 'indeks_kedalaman_p1', \n",
                "            'pengeluaran_per_kapita', 'indeks_pembangunan_manusia_ipm', \n",
                "            'persentase_sanitasi_layak', 'persentase_rumah_layak_huni', 'rasio_cakupan_bansos']\n",
                "df[cols_key].describe().T"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Visualisasi Distribusi Indikator Kemiskinan dan IPM per Pulau\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "sns.boxplot(data=df, x='pulau', y='persentase_penduduk_miskin', palette='Set2', ax=axes[0])\n",
                "axes[0].set_title('Distribusi Persentase Penduduk Miskin (%) per Pulau', fontsize=12, fontweight='bold')\n",
                "axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)\n",
                "\n",
                "sns.boxplot(data=df, x='pulau', y='indeks_pembangunan_manusia_ipm', palette='Pastel1', ax=axes[1])\n",
                "axes[1].set_title('Distribusi Indeks Pembangunan Manusia (IPM) per Pulau', fontsize=12, fontweight='bold')\n",
                "axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Heatmap Korelasi Matriks Indikator Sosioekonomi\n",
                "plt.figure(figsize=(10, 8))\n",
                "corr = df[cols_key].corr()\n",
                "sns.heatmap(corr, annot=True, cmap='RdYlBu', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)\n",
                "plt.title('Matriks Korelasi Indikator Kemiskinan, Infrastruktur & Bansos BPS', fontsize=13, fontweight='bold')\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells_eda, "notebooks/01_eda.ipynb")

    # --- 2. 02_preprocessing.ipynb ---
    cells_prep = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02. Preprocessing & Feature Engineering\n",
                "## Penyusunan Indeks Kemiskinan Multidimensi & Normalisasi Fitur\n",
                "\n",
                "Notebook ini memuat tahapan **Data Preparation**:\n",
                "1. Imputasi dan deteksi outlier berbasis IQR.\n",
                "2. Rekayasa fitur komposit: **Indeks Kemiskinan Multidimensi (IKM)** dan **Indeks Infrastruktur Dasar**.\n",
                "3. Standardisasi fitur dengan `StandardScaler` untuk persiapan input model klasterisasi."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.preprocessing import BPSDataPreprocessor\n",
                "from src.features import compute_bps_regional_features\n",
                "\n",
                "# 1. Load Data\n",
                "df_raw = pd.read_csv('../data/raw/real_bps_kabupaten_kota.csv')\n",
                "\n",
                "# 2. Preprocessing\n",
                "preprocessor = BPSDataPreprocessor(scaler_type='standard')\n",
                "df_clean = preprocessor.clean_and_impute(df_raw)\n",
                "df_clean = preprocessor.handle_outliers_iqr(df_clean)\n",
                "\n",
                "# 3. Feature Engineering\n",
                "df_feat = compute_bps_regional_features(df_clean)\n",
                "print(f'Total Kolom setelah Feature Engineering: {df_feat.shape[1]}')\n",
                "df_feat[['kabupaten_kota', 'indeks_infrastruktur_dasar', 'indeks_kemiskinan_multidimensi_ikm', 'status_alokasi_bansos']].head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Visualisasi Distribusi Indeks Kemiskinan Multidimensi (IKM)\n",
                "plt.figure(figsize=(10, 5))\n",
                "sns.histplot(df_feat['indeks_kemiskinan_multidimensi_ikm'], kde=True, color='darkorange', bins=30)\n",
                "plt.title('Distribusi Indeks Kemiskinan Multidimensi (IKM) 514 Kabupaten/Kota', fontsize=12, fontweight='bold')\n",
                "plt.xlabel('Skor IKM (0: Paling Sejahtera, 100: Paling Miskin/Rentan)')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Standardisasi Fitur untuk Klasterisasi\n",
                "features_for_clustering = [\n",
                "    'persentase_penduduk_miskin',\n",
                "    'garis_kemiskinan',\n",
                "    'indeks_kedalaman_p1',\n",
                "    'indeks_keparahan_p2',\n",
                "    'pengeluaran_per_kapita',\n",
                "    'indeks_pembangunan_manusia_ipm',\n",
                "    'persentase_sanitasi_layak',\n",
                "    'persentase_air_minum_layak',\n",
                "    'persentase_rumah_layak_huni',\n",
                "    'indeks_infrastruktur_dasar',\n",
                "    'indeks_kemiskinan_multidimensi_ikm'\n",
                "]\n",
                "X_scaled, df_scaled = preprocessor.scale_features(df_feat, features_for_clustering)\n",
                "print('Matriks fitur siap klasterisasi dengan bentuk:', X_scaled.shape)"
            ]
        }
    ]
    create_notebook(cells_prep, "notebooks/02_preprocessing.ipynb")

    # --- 3. 03_clustering.ipynb ---
    cells_clust = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 03. Modeling: Klasterisasi & Evaluasi Algoritma\n",
                "## Penentuan k Optimal, K-Means, GMM, Agglomerative & Proyeksi PCA\n",
                "\n",
                "Notebook ini memuat tahapan **Modeling & Evaluation**:\n",
                "1. Penentuan jumlah $k$ optimal menggunakan **Elbow Method (Inertia)** dan **Silhouette Analysis**.\n",
                "2. Melatih dan membandingkan multi-algoritma: **K-Means**, **Gaussian Mixture Model (GMM)**, dan **Agglomerative Hierarchical Clustering**.\n",
                "3. Evaluasi metrik kualitas: **Silhouette Score**, **Davies-Bouldin Index**, dan **Calinski-Harabasz**.\n",
                "4. Reduksi dimensi **PCA 2D dan 3D** untuk memvisualisasikan separasi klaster."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.preprocessing import BPSDataPreprocessor\n",
                "from src.features import compute_bps_regional_features\n",
                "from src.clustering import BPSClusterEngine\n",
                "from src.evaluation import evaluate_k_range, benchmark_clustering_algorithms\n",
                "\n",
                "# Pipeline Persiapan Data\n",
                "df_raw = pd.read_csv('../data/raw/real_bps_kabupaten_kota.csv')\n",
                "p = BPSDataPreprocessor()\n",
                "df_clean = p.clean_and_impute(df_raw)\n",
                "df_clean = p.handle_outliers_iqr(df_clean)\n",
                "df_feat = compute_bps_regional_features(df_clean)\n",
                "\n",
                "clustering_cols = [\n",
                "    'persentase_penduduk_miskin', 'garis_kemiskinan', 'indeks_kedalaman_p1', 'indeks_keparahan_p2',\n",
                "    'pengeluaran_per_kapita', 'indeks_pembangunan_manusia_ipm', 'persentase_sanitasi_layak',\n",
                "    'persentase_air_minum_layak', 'persentase_rumah_layak_huni', 'indeks_infrastruktur_dasar',\n",
                "    'indeks_kemiskinan_multidimensi_ikm'\n",
                "]\n",
                "X_scaled, _ = p.scale_features(df_feat, clustering_cols)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Evaluasi Rentang k (2 sampai 8)\n",
                "k_eval_df = evaluate_k_range(X_scaled, k_min=2, k_max=8)\n",
                "\n",
                "fig, ax1 = plt.subplots(figsize=(10, 5))\n",
                "color = 'tab:blue'\n",
                "ax1.set_xlabel('Jumlah Klaster (k)', fontweight='bold')\n",
                "ax1.set_ylabel('Inertia (Elbow)', color=color, fontweight='bold')\n",
                "ax1.plot(k_eval_df['k'], k_eval_df['inertia'], 'o-', color=color, linewidth=2)\n",
                "ax1.tick_params(axis='y', labelcolor=color)\n",
                "\n",
                "ax2 = ax1.twinx()\n",
                "color = 'tab:red'\n",
                "ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')\n",
                "ax2.plot(k_eval_df['k'], k_eval_df['silhouette_score'], 's--', color=color, linewidth=2)\n",
                "ax2.tick_params(axis='y', labelcolor=color)\n",
                "\n",
                "plt.title('Evaluasi Jumlah Klaster Optimal (Elbow vs Silhouette Score)', fontsize=13, fontweight='bold')\n",
                "plt.show()\n",
                "k_eval_df"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Benchmarking Multi-Algoritma (k=4)\n",
                "benchmark_res = benchmark_clustering_algorithms(X_scaled, n_clusters=4)\n",
                "benchmark_res"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Klasterisasi K-Means dan Proyeksi PCA 2D\n",
                "engine = BPSClusterEngine(n_clusters=4, random_state=42)\n",
                "labels = engine.fit_kmeans(X_scaled)\n",
                "df_feat['cluster'] = labels\n",
                "\n",
                "df_pca_2d, df_pca_3d = engine.compute_pca_projections(X_scaled)\n",
                "df_feat['PCA1'] = df_pca_2d['PCA1']\n",
                "df_feat['PCA2'] = df_pca_2d['PCA2']\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.scatterplot(\n",
                "    data=df_feat, x='PCA1', y='PCA2', hue='cluster', palette='tab10', s=70, alpha=0.9\n",
                ")\n",
                "plt.title('Visualisasi 2D Proyeksi PCA Klaster Kesejahteraan BPS (k=4)', fontsize=13, fontweight='bold')\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells_clust, "notebooks/03_clustering.ipynb")

    # --- 4. 04_targeting_analysis.ipynb ---
    cells_target = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 04. Analisis Ketepatan Sasaran Bansos & Welfare Priority Ranking\n",
                "## Audit Inclusion Error, Exclusion Error & Rekomendasi Alokasi Bantuan\n",
                "\n",
                "Notebook ini memuat tahapan **Evaluation & Deployment**:\n",
                "1. Profiling karakteristik setiap klaster dan auto-labeling.\n",
                "2. Perhitungan **Inclusion Error** dan **Exclusion Error** alokasi bansos.\n",
                "3. Penyusunan **Welfare Priority Index (WPI)** untuk menentukan daftar prioritas intervensi bantuan per daerah se-Indonesia.\n",
                "4. Pelatihan model klasifikasi terawasi (**Random Forest & XGBoost**) untuk memprediksi kelayakan intervensi bantuan sosial."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from src.targeting import (\n",
                "    assign_ordered_cluster_labels, compute_cluster_profiles,\n",
                "    analyze_targeting_errors, compute_welfare_priority_ranking\n",
                ")\n",
                "from src.classification import WelfareClassifierEngine\n",
                "\n",
                "# Load Hasil Pipeline\n",
                "df = pd.read_csv('../data/processed/hasil_audit_bansos_wilayah.csv')\n",
                "print(f'Total data hasil audit: {len(df)} Kabupaten/Kota')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Profil Karakteristik Tiap Klaster\n",
                "profiles = compute_cluster_profiles(df)\n",
                "profiles"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Analisis Ketepatan Sasaran Bansos (Inclusion & Exclusion Error)\n",
                "error_stats = analyze_targeting_errors(df)\n",
                "print('=== HASIL AUDIT SALAH SASARAN BANSOS NASIONAL ===')\n",
                "print(f'Exclusion Error Wilayah : {error_stats[\"exclusion_error_wilayah_pct\"]}%')\n",
                "print(f'Inclusion Error Wilayah : {error_stats[\"inclusion_error_wilayah_pct\"]}%')\n",
                "print('\\nBreakdown Error per Pulau:')\n",
                "error_stats['error_by_island_df']"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Top 15 Daerah Paling Prioritas Berdasarkan Welfare Priority Index (WPI)\n",
                "cols_show = ['ranking_prioritas_bansos', 'kabupaten_kota', 'provinsi', 'klaster_label', \n",
                "             'welfare_priority_index_wpi', 'persentase_penduduk_miskin', 'rasio_cakupan_bansos', 'rekomendasi_kebijakan']\n",
                "df[cols_show].head(15)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Evaluasi Model Klasifikasi Kelayakan Bansos (Random Forest & XGBoost)\n",
                "clustering_features = [\n",
                "    'persentase_penduduk_miskin', 'garis_kemiskinan', 'indeks_kedalaman_p1', 'indeks_keparahan_p2',\n",
                "    'pengeluaran_per_kapita', 'indeks_pembangunan_manusia_ipm', 'persentase_sanitasi_layak',\n",
                "    'persentase_air_minum_layak', 'persentase_rumah_layak_huni', 'indeks_infrastruktur_dasar',\n",
                "    'indeks_kemiskinan_multidimensi_ikm'\n",
                "]\n",
                "\n",
                "clf = WelfareClassifierEngine(random_state=42)\n",
                "X_train, X_test, y_train, y_test = clf.prepare_data(df, feature_cols=clustering_features)\n",
                "clf.train_models(X_train, y_train)\n",
                "eval_dict = clf.evaluate_models(X_test, y_test)\n",
                "\n",
                "print('F1-Score RF     :', eval_dict['Random Forest']['f1_score'])\n",
                "print('F1-Score XGBoost:', eval_dict['XGBoost']['f1_score'])\n",
                "\n",
                "# Feature Importance Ranking\n",
                "imp_df = clf.get_feature_importance()\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(data=imp_df, x='Rata-Rata Importance', y='Fitur', palette='viridis')\n",
                "plt.title('Feature Importance Indikator BPS terhadap Kelayakan Prioritas Bansos', fontsize=12, fontweight='bold')\n",
                "plt.show()\n",
                "imp_df"
            ]
        }
    ]
    create_notebook(cells_target, "notebooks/04_targeting_analysis.ipynb")

if __name__ == "__main__":
    build_all_notebooks()
