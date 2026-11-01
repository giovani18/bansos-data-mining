"""
Modul Klasifikasi Kelayakan Bantuan (Supervised Welfare Classification Engine)
Melatih model Random Forest dan XGBoost untuk memprediksi status kelayakan/prioritas
bantuan sosial berbasis indikator BPS, dilengkapi Feature Importance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, confusion_matrix
import joblib

class WelfareClassifierEngine:
    """
    Model klasifikasi kelayakan prioritas intervensi bantuan sosial.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_model = None
        self.xgb_model = None
        self.feature_names = None
        self.target_names = ["Rendah / Tidak Prioritas", "Tinggi / Prioritas Bansos"]

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list = None,
        target_col: str = None
    ) -> tuple:
        """
        Menyiapkan data training dan testing.
        Target biner: 1 (Prioritas Tinggi Bansos), 0 (Rendah/Tidak Prioritas).
        """
        if feature_cols is None:
            feature_cols = [
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
        self.feature_names = [col for col in feature_cols if col in df.columns]
        
        # Buat label biner target jika belum ada
        if target_col is None or target_col not in df.columns:
            # Wilayah prioritas tinggi jika IKM >= 45.0 atau tergolong Klaster Sangat Miskin/Miskin
            if "klaster_kategori" in df.columns:
                y = df["klaster_kategori"].isin(["Sangat Miskin", "Miskin"]).astype(int).values
            else:
                y = (df["indeks_kemiskinan_multidimensi_ikm"] >= 45.0).astype(int).values
        else:
            y = df[target_col].values

        X = df[self.feature_names].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Melatih Random Forest dan XGBoost Classifier.
        """
        # 1. Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self.random_state
        )
        self.rf_model.fit(X_train, y_train)

        # 2. XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=self.random_state,
            eval_metric="logloss"
        )
        self.xgb_model.fit(X_train, y_train)

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Mengevaluasi F1, Recall, Precision, dan Confusion Matrix.
        """
        results = {}
        for name, model in [("Random Forest", self.rf_model), ("XGBoost", self.xgb_model)]:
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds)
            rec = recall_score(y_test, preds)
            prec = precision_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            report = classification_report(y_test, preds, target_names=self.target_names, output_dict=True)
            
            results[name] = {
                "f1_score": round(f1, 4),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "confusion_matrix": cm,
                "classification_report": report
            }
        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Mengambil skor kepentingan fitur dari XGBoost dan Random Forest.
        """
        df_imp = pd.DataFrame({
            "Fitur": self.feature_names,
            "RF Importance": self.rf_model.feature_importances_,
            "XGBoost Importance": self.xgb_model.feature_importances_
        })
        df_imp["Rata-Rata Importance"] = (df_imp["RF Importance"] + df_imp["XGBoost Importance"]) / 2.0
        df_imp = df_imp.sort_values(by="Rata-Rata Importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save_models(self, rf_path="models/classifier_rf.joblib", xgb_path="models/classifier_xgb.joblib"):
        """
        Menyimpan model ke disk.
        """
        joblib.dump(self.rf_model, rf_path)
        joblib.dump(self.xgb_model, xgb_path)
        print(f"[SUCCESS] Model klasifikasi tersimpan di '{rf_path}' dan '{xgb_path}'.")
