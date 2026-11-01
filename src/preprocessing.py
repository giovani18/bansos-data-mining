"""
Modul Preprocessing Data BPS & Rumah Tangga
Melakukan pembersihan data, penanganan missing values, deteksi & capping outlier,
encoding fitur kategorikal, serta transformasi dan scaling numerik.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BPSDataPreprocessor:
    """
    Preprocessor untuk dataset indikator kemiskinan & bansos BPS Kabupaten/Kota.
    """
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.feature_columns = []
        self.fitted = False

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pembersihan data dan imputasi nilai kosong berbasis median kelompok pulau/provinsi.
        """
        df_clean = df.copy()
        
        # Kolom numerik utama
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                if "pulau" in df_clean.columns:
                    # Imputasi dengan median per pulau
                    df_clean[col] = df_clean.groupby("pulau")[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                # Fallback ke overall median jika masih ada nan
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
        # Kolom kategorikal
        cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
        return df_clean

    def handle_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: list = None,
        factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Melakukan Winsorizing / Capping pada nilai outlier berbasis IQR.
        """
        df_capped = df.copy()
        if columns is None:
            columns = ["pengeluaran_per_kapita", "garis_kemiskinan", "total_penduduk", "alokasi_bansos_per_kapita"]
            
        for col in columns:
            if col in df_capped.columns:
                q1 = df_capped[col].quantile(0.25)
                q3 = df_capped[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (factor * iqr)
                upper_bound = q3 + (factor * iqr)
                df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)
                
        return df_capped

    def scale_features(
        self,
        df: pd.DataFrame,
        feature_cols: list
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Melakukan standardisasi/normalisasi fitur numerik untuk klasterisasi.
        """
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        
        df_scaled = pd.DataFrame(
            X_scaled,
            columns=[f"{col}_scaled" for col in feature_cols],
            index=df.index
        )
        return X_scaled, df_scaled


class HouseholdDataPreprocessor:
    """
    Preprocessor untuk data mikro/sampel rumah tangga Susenas.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
        # Mapping ordinal untuk variabel perumahan & pendidikan
        self.pendidikan_map = {
            "Tidak_Tamat_SD": 0,
            "SD": 1,
            "SMP": 2,
            "SMA_SMK": 3,
            "Diploma_D3": 4,
            "Sarjana_S1": 5,
            "Magister_S2_S3": 6
        }
        self.lantai_map = {
            "Tanah": 0,
            "Semen_Kasar": 1,
            "Ubin_Teraso": 2,
            "Keramik_Sederhana": 3,
            "Keramik_Bagus": 4,
            "Granit_Marmer": 5,
            "Parket_Kayu_Mewah": 5
        }
        self.dinding_map = {
            "Bambu_Anyaman": 0,
            "Kayu_Papan": 1,
            "Tembok_Tanpa_Plester": 2,
            "Tembok_Plester_Biasa": 3,
            "Tembok_Plester_Cat": 4
        }
        self.atap_map = {
            "Rumbia_Ijuk": 0,
            "Seng_Asbes": 1,
            "Genteng_Biasa": 2,
            "Genteng_Pres_Beton": 3,
            "Genteng_Keramik_Mewah": 4
        }
        self.sanitasi_map = {
            "Tidak_Ada_Sungai": 0,
            "Jamban_Cemplung": 1,
            "Jamban_Bersama": 2,
            "Jamban_Leher_Angsa_Bersama": 2,
            "Jamban_Leher_Angsa_Pribadi": 3
        }

    def clean_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mengisi missing values dan melakukan ordinal encoding pada fitur bertingkat.
        """
        df_encoded = df.copy()
        
        # Imputasi missing values
        if "luas_lantai_per_kapita" in df_encoded.columns:
            df_encoded["luas_lantai_per_kapita"] = df_encoded["luas_lantai_per_kapita"].fillna(
                df_encoded["luas_lantai_per_kapita"].median()
            )
        if "pendapatan_per_kapita" in df_encoded.columns:
            df_encoded["pendapatan_per_kapita"] = df_encoded["pendapatan_per_kapita"].fillna(
                df_encoded["pendapatan_per_kapita"].median()
            )
        if "pendidikan_krt" in df_encoded.columns:
            df_encoded["pendidikan_krt"] = df_encoded["pendidikan_krt"].fillna("SD")
            
        # Encoding
        if "pendidikan_krt" in df_encoded.columns:
            df_encoded["pendidikan_score"] = df_encoded["pendidikan_krt"].map(self.pendidikan_map).fillna(1)
        if "jenis_lantai" in df_encoded.columns:
            df_encoded["lantai_score"] = df_encoded["jenis_lantai"].map(self.lantai_map).fillna(1)
        if "jenis_dinding" in df_encoded.columns:
            df_encoded["dinding_score"] = df_encoded["jenis_dinding"].map(self.dinding_map).fillna(1)
        if "jenis_atap" in df_encoded.columns:
            df_encoded["atap_score"] = df_encoded["jenis_atap"].map(self.atap_map).fillna(1)
        if "sanitasi_kloset" in df_encoded.columns:
            df_encoded["sanitasi_score"] = df_encoded["sanitasi_kloset"].map(self.sanitasi_map).fillna(1)
            
        return df_encoded
