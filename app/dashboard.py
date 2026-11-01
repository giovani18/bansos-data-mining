"""
Dashboard Analisis Efektivitas dan Ketepatan Sasaran Bantuan Sosial (Bansos)
Berbasis Data Mining & Data Riil Badan Pusat Statistik (BPS)
Dibangun menggunakan Streamlit & Plotly
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Audit & Data Mining Bansos BPS",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Modern Analytics Theme)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #004D40, #FF6F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card-alert {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-left: 5px solid #E53935;
        margin-bottom: 1rem;
    }
    .metric-card-success {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-left: 5px solid #43A047;
        margin-bottom: 1rem;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1A237E;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #757575;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    processed_path = "data/processed/hasil_audit_bansos_wilayah.csv"
    if not os.path.exists(processed_path):
        from src.pipeline import run_pipeline
        df = run_pipeline(processed_data_path=processed_path)
    else:
        df = pd.read_csv(processed_path)
    return df

df = load_data()

# --- SIDEBAR FILTER ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Lambang_Badan_Pusat_Statistik_%28BPS%29_Indonesia.svg/512px-Lambang_Badan_Pusat_Statistik_%28BPS%29_Indonesia.svg.png", width=70)
st.sidebar.title("🎛️ Filter Analisis")

selected_pulau = st.sidebar.multiselect(
    "Pilih Region Pulau:",
    options=sorted(df["pulau"].unique()),
    default=sorted(df["pulau"].unique())
)

# Filter provinsi berdasarkan pulau terpilih
prov_options = sorted(df[df["pulau"].isin(selected_pulau)]["provinsi"].unique())
selected_prov = st.sidebar.multiselect(
    "Pilih Provinsi:",
    options=prov_options,
    default=[]
)

selected_klaster = st.sidebar.multiselect(
    "Pilih Klaster Kesejahteraan:",
    options=sorted(df["klaster_label"].unique()),
    default=sorted(df["klaster_label"].unique())
)

# Terapkan filter
filtered_df = df[
    (df["pulau"].isin(selected_pulau)) &
    (df["klaster_label"].isin(selected_klaster))
]

if selected_prov:
    filtered_df = filtered_df[filtered_df["provinsi"].isin(selected_prov)]

# --- HEADER UTAMA ---
st.markdown('<div class="main-title">🇮🇩 Dashboard Data Mining: Audit Ketepatan Sasaran Bansos</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analisis Klaster Kesejahteraan 514 Kabupaten/Kota & Evaluasi Efektivitas Alokasi Bantuan Sosial Berbasis Data Riil BPS</div>', unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Wilayah Dianalisis</div>
        <div class="metric-val">{len(filtered_df):,} <span style="font-size:1rem;color:#666;">Kab/Kota</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_p_miskin = filtered_df["persentase_penduduk_miskin"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rata-Rata Penduduk Miskin</div>
        <div class="metric-val">{avg_p_miskin:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    n_miskin_prioritas = filtered_df[filtered_df["klaster_kategori"].isin(["Sangat Miskin", "Miskin"])].shape[0]
    st.markdown(f"""
    <div class="metric-card-alert">
        <div class="metric-label">Daerah Prioritas 1 & 2</div>
        <div class="metric-val">{n_miskin_prioritas} <span style="font-size:1rem;color:#E53935;">Wilayah</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_bansos = filtered_df["total_penerima_bansos"].sum()
    st.markdown(f"""
    <div class="metric-card-success">
        <div class="metric-label">Total Penerima Bansos</div>
        <div class="metric-val">{total_bansos/1e6:.2f}M <span style="font-size:1rem;color:#43A047;">Jiwa</span></div>
    </div>
    """, unsafe_allow_html=True)

# --- TAB NAVIGASI UTAMA ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Ringkasan & Sebaran Data",
    "🎯 Hasil Klaster & PCA",
    "📋 Profil Karakteristik Klaster",
    "⚖️ Audit Salah Sasaran (Error)",
    "🏆 Ranking Prioritas Bansos (WPI)",
    "🧪 Simulator Prediksi Wilayah"
])

# ==========================================
# TAB 1: RINGKASAN & SEBARAN DATA
# ==========================================
with tab1:
    st.subheader("Distribusi Indikator Kemiskinan dan Alokasi Bansos")
    
    col_t1_1, col_t1_2 = st.columns(2)
    with col_t1_1:
        fig_island = px.box(
            filtered_df,
            x="pulau",
            y="persentase_penduduk_miskin",
            color="pulau",
            title="Persentase Penduduk Miskin (%) per Region Pulau",
            labels={"persentase_penduduk_miskin": "Penduduk Miskin (%)", "pulau": "Pulau"},
            points="all",
            hover_data=["kabupaten_kota", "provinsi", "garis_kemiskinan"]
        )
        fig_island.update_layout(showlegend=False)
        st.plotly_chart(fig_island, use_container_width=True)

    with col_t1_2:
        fig_scatter = px.scatter(
            filtered_df,
            x="pengeluaran_per_kapita",
            y="persentase_penduduk_miskin",
            size="jumlah_penduduk_miskin",
            color="klaster_label",
            hover_name="kabupaten_kota",
            hover_data=["provinsi", "indeks_pembangunan_manusia_ipm", "rasio_cakupan_bansos"],
            title="Korelasi Pengeluaran per Kapita vs Persentase Penduduk Miskin",
            labels={"pengeluaran_per_kapita": "Pengeluaran (Rp/Tahun)", "persentase_penduduk_miskin": "% Miskin"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabel Data Terfilter
    with st.expander("🔍 Eksplorasi Data Mentah Kabupaten/Kota Terfilter"):
        st.dataframe(
            filtered_df[[
                "id_kab", "provinsi", "kabupaten_kota", "pulau", "persentase_penduduk_miskin",
                "garis_kemiskinan", "pengeluaran_per_kapita", "indeks_pembangunan_manusia_ipm",
                "indeks_infrastruktur_dasar", "indeks_kemiskinan_multidimensi_ikm", "klaster_label"
            ]],
            use_container_width=True
        )

# ==========================================
# TAB 2: HASIL KLASTERISASI & PROYEKSI PCA
# ==========================================
with tab2:
    st.subheader("Visualisasi Klaster Tingkat Kesejahteraan Wilayah (PCA 2D & 3D)")
    
    col_t2_1, col_t2_2 = st.columns([3, 2])
    with col_t2_1:
        dim_choice = st.radio("Dimensi Proyeksi PCA:", ["2D Scatter Plot", "3D Interactive Space"], horizontal=True)
        
        if dim_choice == "2D Scatter Plot":
            fig_pca = px.scatter(
                filtered_df,
                x="PCA1",
                y="PCA2",
                color="klaster_label",
                hover_name="kabupaten_kota",
                hover_data=["provinsi", "persentase_penduduk_miskin", "indeks_kemiskinan_multidimensi_ikm", "welfare_priority_index_wpi"],
                title="Proyeksi 2D PCA Klaster Kesejahteraan",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_pca.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            fig_pca3d = px.scatter_3d(
                filtered_df,
                x="PCA1",
                y="PCA2",
                z="PCA3",
                color="klaster_label",
                hover_name="kabupaten_kota",
                hover_data=["provinsi", "persentase_penduduk_miskin", "indeks_kemiskinan_multidimensi_ikm"],
                title="Proyeksi 3D PCA Ruang Multidimensi",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_pca3d.update_traces(marker=dict(size=5, opacity=0.85))
            st.plotly_chart(fig_pca3d, use_container_width=True)

    with col_t2_2:
        st.markdown("#### 🏆 Evaluasi Metrik Klasterisasi")
        st.info("""
        **Metrik Kualitas K-Means (k=4):**
        - **Silhouette Score**: `0.335` (Separasi klaster terukur)
        - **Davies-Bouldin Index**: `1.037` (Memenuhi target PRD < 1.20)
        - **Calinski-Harabasz Index**: `519.79` (Kepadatan klaster tinggi)
        """)
        
        # Proporsi Wilayah per Klaster
        cluster_counts = filtered_df["klaster_label"].value_counts().reset_index()
        cluster_counts.columns = ["Klaster", "Jumlah Wilayah"]
        fig_pie = px.pie(
            cluster_counts,
            values="Jumlah Wilayah",
            names="Klaster",
            hole=0.45,
            title="Komposisi Jumlah Daerah per Klaster",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 3: PROFIL KARAKTERISTIK KLASTER
# ==========================================
with tab3:
    st.subheader("Radar Chart & Profil Karakteristik Sosioekonomi Tiap Klaster")
    
    col_t3_1, col_t3_2 = st.columns([3, 2])
    with col_t3_1:
        # Menghitung mean normalisasi 0-100 untuk radar chart
        radar_metrics = [
            "persentase_penduduk_miskin",
            "indeks_kedalaman_p1",
            "indeks_kemiskinan_multidimensi_ikm",
            "indeks_infrastruktur_dasar",
            "persentase_sanitasi_layak",
            "indeks_pembangunan_manusia_ipm"
        ]
        radar_labels = [
            "% Penduduk Miskin",
            "Kedalaman Kemiskinan (P1)",
            "Indeks Kemiskinan Multidimensi (IKM)",
            "Indeks Infrastruktur Dasar",
            "% Sanitasi Layak",
            "IPM"
        ]
        
        fig_radar = go.Figure()
        colors = ["#D32F2F", "#F57C00", "#1976D2", "#388E3C"]
        
        for idx, (klaster_name, group) in enumerate(df.groupby("klaster_label")):
            means = [group[m].mean() for m in radar_metrics]
            # Normalisasi skala radar agar seimbang (0 - 100)
            scaled_means = [
                means[0] * 2.8,  # % miskin skala up
                means[1] * 16.0, # P1 skala up
                means[2],        # IKM (sudah 0-100)
                means[3],        # Infra (sudah 0-100)
                means[4],        # Sanitasi (0-100)
                means[5]         # IPM (0-100)
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=scaled_means,
                theta=radar_labels,
                fill='toself',
                name=klaster_name,
                line=dict(color=colors[idx % len(colors)])
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Perbandingan Multidimensi Antar Klaster (Radar Profil)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_t3_2:
        st.markdown("#### 📖 Narasi Profil Klaster:")
        st.markdown("""
        1. **Klaster 1 (Sangat Miskin / Prioritas 1)**:
           - *Karakteristik*: % Kemiskinan tertinggi (> 20%), infrastruktur dasar sangat tertinggal, IPM < 65.
           - *Fokus Kebijakan*: Bantuan terpadu bansos tunai, sembako, dan pembangunan sanitasi/air bersih.
        2. **Klaster 2 (Miskin / Prioritas 2)**:
           - *Karakteristik*: Kemiskinan di atas rata-rata nasional (10–18%), fasilitas perumahan sedang.
           - *Fokus Kebijakan*: Pemberdayaan PKH berkelanjutan dan subsidi pangan.
        3. **Klaster 3 (Rentan / Berkembang)**:
           - *Karakteristik*: Kemiskinan moderat (5–9%), rentan terhadap guncangan inflasi daya beli.
           - *Fokus Kebijakan*: Penguatan jaring pengaman sosial adaptif.
        4. **Klaster 4 (Sejahtera / Maju)**:
           - *Karakteristik*: Kemiskinan rendah (< 5%), IPM tinggi (> 78), infrastruktur lengkap.
           - *Fokus Kebijakan*: Audit graduasi bansos dan realokasi kuota berlebih.
        """)

# ==========================================
# TAB 4: AUDIT SALAH SASARAN BANSOS
# ==========================================
with tab4:
    st.subheader("Audit Ketepatan Sasaran Bansos (Inclusion & Exclusion Error)")
    
    col_t4_1, col_t4_2 = st.columns(2)
    with col_t4_1:
        # Exclusion Error Gauge
        is_poor = df["klaster_kategori"].isin(["Sangat Miskin", "Miskin"])
        is_under = df["rasio_cakupan_bansos"] < 0.85
        excl_val = round(((is_poor & is_under).sum() / is_poor.sum()) * 100, 1)
        
        fig_gauge1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=excl_val,
            title={'text': "Exclusion Error Wilayah (%)<br><span style='font-size:0.8em;color:gray'>Daerah Miskin yang Defisit Kuota Bansos</span>"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#D32F2F"},
                'steps': [
                    {'range': [0, 20], 'color': "#E8F5E9"},
                    {'range': [20, 40], 'color': "#FFF3E0"},
                    {'range': [40, 100], 'color': "#FFEBEE"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge1, use_container_width=True)

    with col_t4_2:
        # Inclusion Error Gauge
        is_aff = df["klaster_kategori"].isin(["Sejahtera", "Rentan"])
        is_over = df["rasio_cakupan_bansos"] > 1.30
        incl_val = round(((is_aff & is_over).sum() / is_aff.sum()) * 100, 1)
        
        fig_gauge2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=incl_val,
            title={'text': "Inclusion Error Wilayah (%)<br><span style='font-size:0.8em;color:gray'>Daerah Sejahtera yang Surplus/Over-Allocated Bansos</span>"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#F57C00"},
                'steps': [
                    {'range': [0, 15], 'color': "#E8F5E9"},
                    {'range': [15, 30], 'color': "#FFF3E0"},
                    {'range': [30, 100], 'color': "#FFEBEE"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge2, use_container_width=True)

    # Breakdown Ketepatan Alokasi per Pulau
    st.markdown("#### 🗺️ Analisis Ketepatan Alokasi Bansos per Region Pulau")
    fig_bar_error = px.bar(
        filtered_df.groupby(["pulau", "status_alokasi_bansos"]).size().reset_index(name="Jumlah Daerah"),
        x="pulau",
        y="Jumlah Daerah",
        color="status_alokasi_bansos",
        title="Distribusi Status Alokasi Bansos per Pulau",
        barmode="group",
        color_discrete_map={
            "Defisit / Under-Allocated": "#D32F2F",
            "Alokasi Seimbang": "#388E3C",
            "Surplus / Over-Allocated": "#F57C00"
        }
    )
    st.plotly_chart(fig_bar_error, use_container_width=True)

# ==========================================
# TAB 5: RANKING PRIORITAS INTERVENSI BANSOS
# ==========================================
with tab5:
    st.subheader("Daftar Prioritas Intervensi Alokasi Bansos (Welfare Priority Index - WPI)")
    st.write("Daftar 514 Kabupaten/Kota diurutkan berdasarkan skor kerentanan **Welfare Priority Index (WPI)** dari yang paling mendesak (Rank 1).")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_action = st.multiselect(
            "Filter Rekomendasi Kebijakan:",
            options=list(df["rekomendasi_kebijakan"].unique()),
            default=[]
        )
    with col_f2:
        search_query = st.text_input("🔍 Cari Nama Kabupaten/Kota atau Provinsi:", "")
        
    display_df = filtered_df.sort_values(by="ranking_prioritas_bansos")
    if filter_action:
        display_df = display_df[display_df["rekomendasi_kebijakan"].isin(filter_action)]
    if search_query:
        display_df = display_df[
            display_df["kabupaten_kota"].str.contains(search_query, case=False) |
            display_df["provinsi"].str.contains(search_query, case=False)
        ]
        
    st.dataframe(
        display_df[[
            "ranking_prioritas_bansos", "kabupaten_kota", "provinsi", "pulau",
            "welfare_priority_index_wpi", "persentase_penduduk_miskin", "rasio_cakupan_bansos",
            "klaster_label", "rekomendasi_kebijakan"
        ]],
        use_container_width=True,
        height=400
    )
    
    # Tombol Download CSV
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Hasil Audit & Ranking Bansos (CSV)",
        data=csv_data,
        file_name="hasil_audit_prioritas_bansos_bps.csv",
        mime="text/csv"
    )

# ==========================================
# TAB 6: SIMULATOR PREDIKSI WILAYAH BARU
# ==========================================
with tab6:
    st.subheader("🧪 Simulator Prediksi Klaster & Kelayakan Bantuan Sosial")
    st.write("Masukkan indikator sosioekonomi suatu daerah atau desa untuk memprediksi klaster kesejahteraan dan rekomendasi alokasi secara instan.")
    
    with st.form("sim_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            in_p_miskin = st.number_input("Persentase Penduduk Miskin (%)", min_value=1.0, max_value=50.0, value=12.5)
            in_gk = st.number_input("Garis Kemiskinan (Rp/kapita/bulan)", min_value=300000, max_value=1200000, value=550000, step=25000)
            in_p1 = st.number_input("Indeks Kedalaman Kemiskinan (P1)", min_value=0.1, max_value=8.0, value=1.8)
        with col_s2:
            in_pengeluaran = st.number_input("Pengeluaran per Kapita (Rp/tahun)", min_value=4000000, max_value=30000000, value=11000000, step=500000)
            in_ipm = st.number_input("Indeks Pembangunan Manusia (IPM)", min_value=45.0, max_value=90.0, value=71.5)
            in_sanitasi = st.number_input("% Akses Sanitasi Layak", min_value=20.0, max_value=100.0, value=75.0)
        with col_s3:
            in_air = st.number_input("% Akses Air Minum Layak", min_value=20.0, max_value=100.0, value=80.0)
            in_rumah = st.number_input("% Rumah Layak Huni", min_value=20.0, max_value=100.0, value=78.0)
            in_rasio_bansos = st.number_input("Rasio Cakupan Bansos Aktual terhadap Kebutuhan", min_value=0.2, max_value=2.5, value=0.75)
            
        submitted = st.form_submit_button("🚀 Jalankan Prediksi Klaster & Rekomendasi")
        
    if submitted:
        # Hitung fitur turunan
        infra_sim = 0.30 * in_sanitasi + 0.30 * in_air + 0.25 * in_rumah + 0.15 * 95.0
        p0_n = np.clip(in_p_miskin / 35.0, 0, 1)
        p1_n = np.clip(in_p1 / 6.0, 0, 1)
        ipm_n = np.clip((88.0 - in_ipm) / 40.0, 0, 1)
        infra_n = np.clip((100.0 - infra_sim) / 65.0, 0, 1)
        ikm_sim = (0.35 * p0_n + 0.20 * p1_n + 0.25 * ipm_n + 0.20 * infra_n) * 100.0
        
        # WPI Score
        wpi_sim = (
            0.45 * (ikm_sim / 100.0) +
            0.35 * np.clip((1.5 - in_rasio_bansos) / 1.5, 0, 1) +
            0.20 * np.clip(((in_gk * 12) / in_pengeluaran) / 1.2, 0, 1)
        ) * 100.0
        
        # Penentuan Klaster
        if ikm_sim >= 60.0:
            pred_klaster = "Klaster 1: Sangat Miskin / Prioritas 1"
            badge_color = "#D32F2F"
        elif ikm_sim >= 42.0:
            pred_klaster = "Klaster 2: Miskin / Prioritas 2"
            badge_color = "#F57C00"
        elif ikm_sim >= 25.0:
            pred_klaster = "Klaster 3: Rentan / Berkembang"
            badge_color = "#1976D2"
        else:
            pred_klaster = "Klaster 4: Sejahtera / Maju"
            badge_color = "#388E3C"
            
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi Simulasi:")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(f"""
            <div style="background:#f0f4f8;padding:1.2rem;border-radius:10px;border-left:5px solid {badge_color}">
                <div style="color:#666;font-size:0.85rem;font-weight:bold;">PREDIKSI KLASTER:</div>
                <div style="font-size:1.3rem;font-weight:800;color:{badge_color}">{pred_klaster}</div>
            </div>
            """, unsafe_allow_html=True)
        with res_col2:
            st.markdown(f"""
            <div style="background:#f0f4f8;padding:1.2rem;border-radius:10px;border-left:5px solid #1E88E5">
                <div style="color:#666;font-size:0.85rem;font-weight:bold;">SKOR IKM MULTIDIMENSI:</div>
                <div style="font-size:1.4rem;font-weight:800;color:#1E88E5">{ikm_sim:.2f} / 100</div>
            </div>
            """, unsafe_allow_html=True)
        with res_col3:
            st.markdown(f"""
            <div style="background:#f0f4f8;padding:1.2rem;border-radius:10px;border-left:5px solid #7B1FA2">
                <div style="color:#666;font-size:0.85rem;font-weight:bold;">WELFARE PRIORITY INDEX (WPI):</div>
                <div style="font-size:1.4rem;font-weight:800;color:#7B1FA2">{wpi_sim:.2f} / 100</div>
            </div>
            """, unsafe_allow_html=True)
            
        if wpi_sim >= 60.0:
            st.error("🚨 **Rekomendasi Kebijakan:** Wilayah ini tergolong **Prioritas Utama (Mendesak)** untuk penambahan kuota bansos dan pembangunan infrastruktur dasar.")
        elif wpi_sim >= 40.0:
            st.warning("⚠️ **Rekomendasi Kebijakan:** Wilayah ini memerlukan monitoring intensif dan intervensi bansos bersyarat (PKH).")
        else:
            st.success("✅ **Rekomendasi Kebijakan:** Wilayah berkategori aman/berkembang. Pertahankan alokasi atau lakukan graduasi bansos mandiri.")
