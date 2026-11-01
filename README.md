# Analisis Efektivitas dan Ketepatan Sasaran Bantuan Sosial (Bansos) Berbasis Data Riil BPS

Proyek Data Mining komprehensif berbasis metodologi **CRISP-DM** untuk mengelompokkan tingkat kesejahteraan dan kemiskinan wilayah 514 Kabupaten/Kota di 38 Provinsi di Indonesia berdasarkan data riil **Badan Pusat Statistik (BPS)** dan **Satu Data Indonesia**, mengevaluasi ketepatan alokasi bantuan sosial (*Inclusion Error* dan *Exclusion Error*), menyusun *Welfare Priority Index (WPI)*, serta menyajikan dashboard analitik interaktif.

---

## 📌 Ringkasan Masalah & Tujuan

Penyaluran bantuan sosial (seperti PKH, BPNT/Sembako, BLT, PBI-JK) sering kali menghadapi tantangan disparitas antarwilayah:
1. **Inclusion Error**: Wilayah/kelompok mampu yang menerima alokasi bansos berlebih di luar proporsi kebutuhan.
2. **Exclusion Error**: Wilayah/kelompok sangat miskin atau rentan yang mengalami defisit cakupan bansos (*under-coverage*).

**Tujuan Proyek:**
- Membentuk klaster kesejahteraan objektif (*Sangat Miskin / Prioritas 1, Miskin / Prioritas 2, Rentan / Berkembang, Sejahtera / Maju*) berbasis data riil BPS (Garis Kemiskinan, Pengeluaran, Sanitasi, Air Bersih, Kelayakan Rumah, IPM).
- Menghitung estimasi *Inclusion Error* dan *Exclusion Error* secara nasional dan per provinsi/wilayah.
- Menghasilkan daftar prioritas intervensi alokasi bansos berbasis *Welfare Priority Index (WPI)*.
- Membangun dashboard interaktif berbasis Streamlit dengan visualisasi peta, profil klaster, dan simulator alokasi.

---

## 🏗️ Struktur Direktori

```text
bansos-mining/
├── data/
│   ├── raw/                   # Dataset hasil scraping/ingestion data riil BPS
│   │   └── real_bps_kabupaten_kota.csv
│   └── processed/             # Dataset bersih dan hasil skor prioritas
│       └── hasil_audit_bansos_wilayah.csv
├── notebooks/                 # Jupyter Notebooks langkah demi langkah
│   ├── 01_eda.ipynb           # EDA, distribusi variabel BPS, korelasi kemiskinan
│   ├── 02_preprocessing.ipynb # Data cleaning, feature engineering indeks kemiskinan
│   ├── 03_clustering.ipynb    # K-Means, GMM, Agglomerative, evaluasi Silhouette/Davies-Bouldin
│   └── 04_targeting_analysis.ipynb # Inclusion/exclusion error, ranking prioritas alokasi bansos
├── src/                       # Modul Python
│   ├── scrape_real_data.py    # Modul web scraping & ingestion data resmi BPS
│   ├── preprocessing.py       # Imputasi, encoding, outlier handling, scaling
│   ├── features.py            # Feature engineering (Indeks Kemiskinan Multidimensi & Gap Bansos)
│   ├── clustering.py          # Implementasi algoritma clustering & PCA 2D/3D
│   ├── evaluation.py          # Metrik evaluasi (Silhouette, Davies-Bouldin, Calinski, Elbow)
│   ├── targeting.py           # Perhitungan error bansos, profil, Welfare Priority Index
│   ├── classification.py      # Model klasifikasi kelayakan prioritas (XGBoost / Random Forest)
│   └── pipeline.py            # CLI orkestrator pipeline end-to-end
├── app/
│   └── dashboard.py           # Dashboard Streamlit interaktif
├── models/                    # Model machine learning tersimpan (.joblib)
├── reports/                   # Ringkasan ekspor laporan & visualisasi
├── requirements.txt           # Dependensi Python
├── prd.md                     # Dokumen PRD resmi berbasis data riil BPS
└── README.md                  # Dokumentasi proyek
```

---

## 🚀 Panduan Menjalankan Proyek

### 1. Persiapan Environment
Gunakan virtual environment Python 3.11+:
```bash
# Menjalankan virtual environment
.venv\Scripts\activate   # di Windows PowerShell

# Memasang dependensi
pip install -r requirements.txt
```

### 2. Mengambil / Memperbarui Data Riil BPS (Scraping & Ingestion)
```bash
python src/scrape_real_data.py
```

### 3. Menjalankan Pipeline Lengkap (End-to-End)
```bash
python -m src.pipeline
```

### 4. Menjalankan Dashboard Interaktif Streamlit
```bash
streamlit run app/dashboard.py
```

---

## 📊 Metodologi CRISP-DM

| Tahap | Aktivitas dalam Proyek |
|---|---|
| **1. Business Understanding** | Definisi kriteria kemiskinan BPS, perumusan *inclusion* dan *exclusion error* alokasi bansos. |
| **2. Data Understanding** | Scraping & EDA data 514 Kabupaten/Kota se-Indonesia, analisis korelasi IPM, sanitasi, dan bansos. |
| **3. Data Preparation** | Penanganan outlier, rekayasa fitur Indeks Kemiskinan Multidimensi (IKM) & Rasio Cakupan Bansos. |
| **4. Modeling** | Klasterisasi K-Means, GMM, Agglomerative; optimasi jumlah $k$ (Elbow curve & Silhouette Analysis). |
| **5. Evaluation** | Silhouette Score ($\ge 0.40$), Davies-Bouldin Index ($< 1.20$), validasi profil sosioekonomi per pulau. |
| **6. Deployment** | Ekspor CSV hasil audit dan Dashboard visualisasi interaktif Streamlit. |

---

## 📈 Indikator Resmi BPS yang Digunakan

1. **Moneter**: Persentase Penduduk Miskin (P0), Garis Kemiskinan (GK), Indeks Kedalaman Kemiskinan (P1), Indeks Keparahan Kemiskinan (P2), Pengeluaran per Kapita Disesuaikan.
2. **Kualitas Hidup & Perumahan (Susenas)**: Indeks Pembangunan Manusia (IPM), % Akses Sanitasi Layak, % Akses Air Minum Layak, % Rumah Layak Huni, % Akses Listrik.
3. **Bantuan Sosial**: Penerima PKH, Penerima Sembako/BPNT, Alokasi Bansos per Kapita, Rasio Cakupan Bansos.
