"""
Modul Rekayasa Fitur (Feature Engineering) Data BPS & Rumah Tangga
Menghasilkan Indeks Kemiskinan Multidimensi (IKM), Indeks Infrastruktur Dasar,
Rasio Gap Bansos Wilayah, serta Indeks Kerentanan Kesejahteraan.
"""

import numpy as np
import pandas as pd

def compute_bps_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung fitur turunan dan indeks komposit untuk data BPS Kabupaten/Kota.
    """
    df_feat = df.copy()
    
    # 1. Indeks Infrastruktur & Fasilitas Dasar (0 - 100, semakin tinggi semakin baik)
    df_feat["indeks_infrastruktur_dasar"] = (
        0.30 * df_feat["persentase_sanitasi_layak"] +
        0.30 * df_feat["persentase_air_minum_layak"] +
        0.25 * df_feat["persentase_rumah_layak_huni"] +
        0.15 * df_feat["persentase_akses_listrik"]
    ).round(2)
    
    # 2. Rasio Kemiskinan terhadap Daya Beli (Garis Kemiskinan Tahunan / Pengeluaran Riil Tahunan)
    # Semakin tinggi nilainya, semakin rentan daerah tersebut
    df_feat["rasio_beban_garis_kemiskinan"] = (
        (df_feat["garis_kemiskinan"] * 12) / df_feat["pengeluaran_per_kapita"]
    ).round(3)

    # 3. Indeks Kemiskinan Multidimensi (IKM) BPS (Skala 0 - 100)
    # Skor 100: Daerah Sangat Miskin Ekstrem, Skor 0: Daerah Sangat Sejahtera
    # Komponen:
    # - Dimensi Moneter: P0 (Persentase Miskin) & P1 (Kedalaman)
    # - Dimensi Kapabilitas: Inversi IPM (100 - IPM)
    # - Dimensi Deprivasi Fisik: Inversi Infrastruktur Dasar (100 - Infrastruktur)
    p0_norm = np.clip(df_feat["persentase_penduduk_miskin"] / 35.0, 0, 1)
    p1_norm = np.clip(df_feat["indeks_kedalaman_p1"] / 6.0, 0, 1)
    ipm_inv_norm = np.clip((88.0 - df_feat["indeks_pembangunan_manusia_ipm"]) / 40.0, 0, 1)
    infra_inv_norm = np.clip((100.0 - df_feat["indeks_infrastruktur_dasar"]) / 65.0, 0, 1)

    ikm_score = (
        0.35 * p0_norm +
        0.20 * p1_norm +
        0.25 * ipm_inv_norm +
        0.20 * infra_inv_norm
    ) * 100.0
    
    df_feat["indeks_kemiskinan_multidimensi_ikm"] = ikm_score.round(2)

    # 4. Fitur Audit Bantuan Sosial (Targeting Error Features)
    # Estimasi kuota kebutuhan objektif bansos
    df_feat["estimasi_kebutuhan_bansos"] = (df_feat["jumlah_penduduk_miskin"] * 1.15).astype(int)
    
    # Gap Alokasi Bansos (Positif: Surplus / Over-allocated, Negatif: Defisit / Under-allocated)
    df_feat["gap_kuota_bansos"] = df_feat["total_penerima_bansos"] - df_feat["estimasi_kebutuhan_bansos"]
    
    # Kategori Ketepatan Alokasi Wilayah
    # Rasio > 1.30: Over-Allocated (Potensi Inclusion Error Wilayah)
    # Rasio 0.85 - 1.30: Alokasi Seimbang / Terarah
    # Rasio < 0.85: Under-Allocated (Potensi Exclusion Error Wilayah)
    def categorize_allocation(rasio):
        if rasio > 1.30:
            return "Surplus / Over-Allocated"
        elif rasio < 0.85:
            return "Defisit / Under-Allocated"
        else:
            return "Alokasi Seimbang"

    df_feat["status_alokasi_bansos"] = df_feat["rasio_cakupan_bansos"].apply(categorize_allocation)

    return df_feat


def compute_household_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung indeks komposit untuk tingkat rumah tangga.
    """
    df_feat = df.copy()

    # 1. Indeks Kondisi Rumah (0 - 100)
    if "lantai_score" in df_feat.columns and "dinding_score" in df_feat.columns:
        lantai_norm = df_feat["lantai_score"] / 5.0
        dinding_norm = df_feat["dinding_score"] / 4.0
        atap_norm = df_feat["atap_score"] / 4.0
        luas_norm = np.clip(df_feat["luas_lantai_per_kapita"] / 30.0, 0, 1)
        
        df_feat["indeks_kondisi_rumah"] = (
            (0.30 * lantai_norm + 0.30 * dinding_norm + 0.20 * atap_norm + 0.20 * luas_norm) * 100
        ).round(2)

    # 2. Indeks Aset (0 - 100)
    if "aset_motor" in df_feat.columns:
        aset_score = (
            0.25 * np.clip(df_feat["aset_motor"], 0, 2) / 2.0 +
            0.40 * np.clip(df_feat["aset_mobil"], 0, 1) +
            0.20 * np.clip(df_feat["aset_kulkas"], 0, 1) +
            0.15 * np.clip(df_feat["aset_tv"], 0, 1)
        ) * 100.0
        df_feat["indeks_aset"] = aset_score.round(2)

    # 3. Rasio Ketergantungan (Dependency Ratio)
    if "jml_art" in df_feat.columns:
        non_produktif = df_feat["jml_balita"] + df_feat["jml_lansia"] + df_feat["jml_disabilitas"]
        produktif = np.maximum(1, df_feat["jml_art"] - non_produktif)
        df_feat["rasio_ketergantungan"] = (non_produktif / produktif).round(2)

    # 4. Rasio Pengeluaran terhadap Garis Kemiskinan Wilayah
    if "pengeluaran_per_kapita" in df_feat.columns and "garis_kemiskinan_wilayah" in df_feat.columns:
        df_feat["rasio_pengeluaran_gk"] = (
            df_feat["pengeluaran_per_kapita"] / df_feat["garis_kemiskinan_wilayah"]
        ).round(2)

    return df_feat
