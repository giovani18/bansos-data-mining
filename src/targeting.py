"""
Modul Analisis Ketepatan Sasaran (Targeting Analysis & Welfare Priority Scoring)
Melakukan profiling klaster, pelabelan otomatis berbasis kemiskinan BPS,
penghitungan Inclusion & Exclusion Error alokasi bansos, serta Welfare Priority Index (WPI).
"""

import numpy as np
import pandas as pd

def assign_ordered_cluster_labels(
    df: pd.DataFrame,
    cluster_col: str = "cluster_raw",
    ranking_metric: str = "indeks_kemiskinan_multidimensi_ikm"
) -> tuple[pd.DataFrame, dict]:
    """
    Memberikan label deskriptif yang terurut secara konsisten berdasarkan
    tingkat kemiskinan/kerentanan (Klaster 1: Paling Miskin s.d. Klaster 4: Paling Sejahtera).
    """
    df_labeled = df.copy()
    
    # Urutkan klaster berdasarkan rata-rata tingkat kemiskinan tertinggi ke terendah
    cluster_means = df_labeled.groupby(cluster_col)[ranking_metric].mean().sort_values(ascending=False)
    
    label_names = [
        "Klaster 1: Sangat Miskin / Prioritas 1",
        "Klaster 2: Miskin / Prioritas 2",
        "Klaster 3: Rentan / Berkembang",
        "Klaster 4: Sejahtera / Maju"
    ]
    
    label_short = [
        "Sangat Miskin",
        "Miskin",
        "Rentan",
        "Sejahtera"
    ]
    
    label_mapping = {}
    label_short_mapping = {}
    
    for rank, orig_cluster in enumerate(cluster_means.index):
        label_mapping[orig_cluster] = label_names[rank] if rank < len(label_names) else f"Klaster {rank+1}"
        label_short_mapping[orig_cluster] = label_short[rank] if rank < len(label_short) else f"K-{rank+1}"
        
    df_labeled["klaster_label"] = df_labeled[cluster_col].map(label_mapping)
    df_labeled["klaster_kategori"] = df_labeled[cluster_col].map(label_short_mapping)
    df_labeled["klaster_id"] = df_labeled[cluster_col].map(lambda x: list(cluster_means.index).index(x) + 1)
    
    return df_labeled, label_mapping


def compute_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung ringkasan statistik dan profil sosioekonomi per klaster.
    """
    cols_to_profile = [
        "persentase_penduduk_miskin",
        "garis_kemiskinan",
        "indeks_kedalaman_p1",
        "pengeluaran_per_kapita",
        "indeks_pembangunan_manusia_ipm",
        "persentase_sanitasi_layak",
        "persentase_air_minum_layak",
        "persentase_rumah_layak_huni",
        "indeks_infrastruktur_dasar",
        "indeks_kemiskinan_multidimensi_ikm",
        "rasio_cakupan_bansos"
    ]
    
    valid_cols = [c for c in cols_to_profile if c in df.columns]
    
    profile_df = df.groupby("klaster_label")[valid_cols].agg(["mean", "median"]).round(2)
    profile_df["jumlah_wilayah"] = df.groupby("klaster_label").size()
    profile_df["proporsi_wilayah"] = (df.groupby("klaster_label").size() / len(df) * 100).round(1)
    
    return profile_df


def analyze_targeting_errors(df: pd.DataFrame) -> dict:
    """
    Menghitung metrik Inclusion Error dan Exclusion Error alokasi bantuan sosial:
    - Inclusion Error: Alokasi surplus/berlebih pada daerah/kelompok Sejahtera (Klaster 4)
    - Exclusion Error: Defisit cakupan bansos pada daerah Sangat Miskin & Miskin (Klaster 1 & 2)
    """
    total_wilayah = len(df)
    
    # 1. Klasifikasi Miskin vs Non-Miskin berdasarkan Klaster Objektif
    is_poor_cluster = df["klaster_kategori"].isin(["Sangat Miskin", "Miskin"])
    is_affluent_cluster = df["klaster_kategori"].isin(["Sejahtera", "Rentan"])
    
    # 2. Status Alokasi Bansos
    is_under_covered = df["rasio_cakupan_bansos"] < 0.85
    is_over_allocated = df["rasio_cakupan_bansos"] > 1.30
    
    # Exclusion Error Wilayah: Daerah Sangat Miskin/Miskin yang defisit bansos
    n_poor_total = is_poor_cluster.sum()
    n_poor_excluded = (is_poor_cluster & is_under_covered).sum()
    exclusion_error_pct = round((n_poor_excluded / max(1, n_poor_total)) * 100, 2)
    
    # Inclusion Error Wilayah: Daerah Sejahtera/Rentan yang over-allocated bansos
    n_affluent_total = is_affluent_cluster.sum()
    n_affluent_included = (is_affluent_cluster & is_over_allocated).sum()
    inclusion_error_pct = round((n_affluent_included / max(1, n_affluent_total)) * 100, 2)
    
    # Error breakdown per pulau
    error_by_island = []
    if "pulau" in df.columns:
        for pulau, group in df.groupby("pulau"):
            g_poor = group["klaster_kategori"].isin(["Sangat Miskin", "Miskin"])
            g_excl = (g_poor & (group["rasio_cakupan_bansos"] < 0.85)).sum()
            g_excl_pct = round((g_excl / max(1, g_poor.sum())) * 100, 1) if g_poor.sum() > 0 else 0.0
            
            g_aff = group["klaster_kategori"].isin(["Sejahtera", "Rentan"])
            g_incl = (g_aff & (group["rasio_cakupan_bansos"] > 1.30)).sum()
            g_incl_pct = round((g_incl / max(1, g_aff.sum())) * 100, 1) if g_aff.sum() > 0 else 0.0
            
            error_by_island.append({
                "Pulau": pulau,
                "Total Wilayah": len(group),
                "Jumlah Miskin/Prioritas": g_poor.sum(),
                "Exclusion Error (%)": g_excl_pct,
                "Inclusion Error (%)": g_incl_pct
            })
            
    df_error_island = pd.DataFrame(error_by_island)
    
    return {
        "total_wilayah": total_wilayah,
        "n_miskin_wilayah": int(n_poor_total),
        "n_sejahtera_wilayah": int(n_affluent_total),
        "exclusion_error_wilayah_pct": exclusion_error_pct,
        "inclusion_error_wilayah_pct": inclusion_error_pct,
        "error_by_island_df": df_error_island
    }


def compute_welfare_priority_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung Welfare Priority Index (WPI) skala 0 - 100 dan menyusun ranking prioritas bantuan.
    Semakin tinggi WPI (Rank 1), semakin mendesak untuk menerima tambahan alokasi bansos.
    """
    df_ranked = df.copy()
    
    # Komponen Skor Prioritas:
    # 1. Indeks Kemiskinan Multidimensi (45%)
    # 2. Defisit Rasio Cakupan Bansos (30%) -> Semakin kecil rasio cakupan, semakin tinggi prioritas
    # 3. Rasio Beban Garis Kemiskinan terhadap Pengeluaran (25%)
    
    ikm_norm = np.clip(df_ranked["indeks_kemiskinan_multidimensi_ikm"] / 100.0, 0, 1)
    
    # Inversi rasio cakupan (jika rasio rendah < 0.85, skor tinggi)
    cakupan_deficit_norm = np.clip((1.5 - df_ranked["rasio_cakupan_bansos"]) / 1.5, 0, 1)
    
    beban_gk_norm = np.clip(df_ranked["rasio_beban_garis_kemiskinan"] / 1.2, 0, 1)
    
    wpi_score = (
        0.45 * ikm_norm +
        0.35 * cakupan_deficit_norm +
        0.20 * beban_gk_norm
    ) * 100.0
    
    df_ranked["welfare_priority_index_wpi"] = wpi_score.round(2)
    
    # Urutkan dari WPI tertinggi (paling butuh)
    df_ranked = df_ranked.sort_values(by="welfare_priority_index_wpi", ascending=False).reset_index(drop=True)
    df_ranked["ranking_prioritas_bansos"] = df_ranked.index + 1
    
    # Rekomendasi Kebijakan Alokasi
    def recommend_action(row):
        wpi = row["welfare_priority_index_wpi"]
        rasio = row["rasio_cakupan_bansos"]
        if wpi >= 65.0 and rasio < 0.90:
            return "Intervensi Mendesak: Tambah Kuota Bansos (Top Priority)"
        elif wpi >= 50.0:
            return "Perlu Peningkatan Alokasi & Verifikasi Lapangan"
        elif wpi <= 30.0 and rasio > 1.30:
            return "Audit & Realokasi Kuota Berlebih (Over-Targeting)"
        else:
            return "Pertahankan Alokasi / Monitoring Berkala"
            
    df_ranked["rekomendasi_kebijakan"] = df_ranked.apply(recommend_action, axis=1)
    
    return df_ranked
