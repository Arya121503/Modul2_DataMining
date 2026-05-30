# ============================================================
# SCRIPT OTOMASI PRAKTIKUM: DATA MINING & DEPLOYMENT
# ============================================================
# File ini menjalankan seluruh alur praktikum secara berurutan:
# - Modul 2: Step 1 sampai Step 6 (Data Loading s/d Evaluation)
# - Modul 3: Step 7 (Penyimpanan Model & Metadata untuk Web Deployment)

import sys
sys.path.insert(0, '.')
# Pastikan terminal menggunakan UTF-8 agar tidak terjadi UnicodeEncodeError pada Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import ModelBuilder
from src.evaluation import ModelEvaluator
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("             ALUR OTOMATIS: MODUL 2 & MODUL 3")
print("="*60)

# ============================================================
# STEP 1: DATA LOADING
# ============================================================
print("\n[STEP 1] LOADING DATA")
print("-"*60)

loader = DataLoader("data/raw")

dataset_candidates = ["customer_purchase_data.csv", "dataset.csv"]
dataset_filename = next((name for name in dataset_candidates if (Path("data/raw") / name).exists()), None)

if dataset_filename is not None:
    dataset_path = Path("data/raw") / dataset_filename
    print(f"📥 Memuat dataset dari: {dataset_path.as_posix()}")
    df = loader.load_csv(dataset_filename)
else:
    print("📝 Dataset tidak ditemukan, membuat sample dataset untuk demo...")
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,
        'income': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000] * 10,
        'education': ['High School', 'Bachelor', 'Master', 'PhD', 'High School'] * 20,
        'experience': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19] * 10,
        'purchased': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 10
    })

loader.get_data_info(df)

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n[STEP 2] EXPLORATORY DATA ANALYSIS")
print("-"*60)

print("📊 Data Summary:")
print(df.describe())

print("\n🔍 Data Types:")
print(df.dtypes)

print("\n⚠️ Missing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================
print("\n[STEP 3] DATA PREPROCESSING")
print("-"*60)

preprocessor = DataPreprocessor()

# Handle missing values (jika ada)
df_cleaned = preprocessor.handle_missing_values(df, strategy='mean')

# Encode categorical variables
categorical_cols = df_cleaned.select_dtypes(include=['object', 'string']).columns.tolist()
df_cleaned = preprocessor.encode_categorical(df_cleaned, categorical_cols)

print("✓ Data preprocessing selesai!")
print(df_cleaned.head())

# ============================================================
# STEP 4: PREPARE FEATURES & TARGET
# ============================================================
print("\n[STEP 4] PREPARE FEATURES & TARGET")
print("-"*60)

preferred_targets = ["purchased", "loan_approved", "target", "label"]
target_col = next((c for c in preferred_targets if c in df_cleaned.columns), None)
if target_col is None:
    target_col = df_cleaned.columns[-1]
    print(f"⚠️ Target column tidak ditemukan di daftar default; memakai kolom terakhir: {target_col}")
else:
    print(f"✓ Target column: {target_col}")

X = df_cleaned.drop(target_col, axis=1)
y = df_cleaned[target_col]

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Feature names: {list(X.columns)}")

# ============================================================
# STEP 5: MODEL BUILDING
# ============================================================
print("\n[STEP 5] MODEL BUILDING")
print("-"*60)

builder = ModelBuilder()

# Split data
X_train, X_test, y_train, y_test = builder.split_data(X, y, test_size=0.2)

# Normalize/Scale data
preprocessor_scaled = DataPreprocessor()
X_train_scaled = preprocessor_scaled.normalize_scale(X_train, X_train.columns, method='standard', fit=True)
X_test_scaled = preprocessor_scaled.normalize_scale(X_test, X_test.columns, method='standard', fit=False)

# Train multiple models
print("\n🤖 Training models...")
model_lr = builder.train_logistic_regression(X_train_scaled, y_train)
model_rf = builder.train_random_forest(X_train_scaled, y_train, n_estimators=100)

# ============================================================
# STEP 6: MODEL EVALUATION
# ============================================================
print("\n[STEP 6] MODEL EVALUATION")
print("-"*60)

evaluator = ModelEvaluator()

# Get predictions
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_rf = model_rf.predict(X_test_scaled)

# Evaluate
print("\n📊 Logistic Regression Evaluation:")
metrics_lr = evaluator.evaluate_classification(y_test, y_pred_lr)

print("\n📊 Random Forest Evaluation:")
metrics_rf = evaluator.evaluate_classification(y_test, y_pred_rf)

# Compare models
print("\n📊 Model Comparison:")
comparison = evaluator.compare_models(
    {'Logistic Regression': model_lr, 'Random Forest': model_rf},
    X_test_scaled, y_test
)

# ============================================================
# MODUL 3: PREPARING MODEL FOR DEPLOYMENT (SERIALIZATION)
# ============================================================
print("\n" + "="*60)
print("     MODUL 3: PENYIAPAN MODEL UNTUK DEPLOYMENT (SERIALISASI)")
print("="*60)
print("\n[STEP 7] SAVE MODEL & METADATA ARTIFACTS")
print("-"*60)

# Pastikan folder output tersedia sebelum menyimpan hasil
Path("results/models").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save best model
best_model = model_rf if metrics_rf['accuracy'] > metrics_lr['accuracy'] else model_lr
builder.save_model(best_model, 'results/models/best_model.pkl')

# Save preprocessors and metadata for deployment
builder.save_model(preprocessor, 'results/models/preprocessor.pkl')
builder.save_model(preprocessor_scaled, 'results/models/scaler.pkl')

# Save features metadata for UI generation
feature_metadata = {
    'features': list(X.columns),
    'target': target_col,
    'details': {}
}
for col in X.columns:
    if col in categorical_cols:
        le = preprocessor.label_encoders[col]
        feature_metadata['details'][col] = {
            'type': 'categorical',
            'categories': list(le.classes_)
        }
    else:
        feature_metadata['details'][col] = {
            'type': 'numeric',
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean())
        }
builder.save_model(feature_metadata, 'results/models/feature_metadata.pkl')

# Save processed data
df_cleaned.to_csv('data/processed/dataset_clean.csv', index=False)
print("✓ Data processed disimpan: data/processed/dataset_clean.csv")

# ============================================================
# SUMMARY PRAKTIKUM
# ============================================================
print("\n" + "="*60)
print("🎉 PROSES MODUL 2 & MODUL 3 SELESAI!")
print("="*60)
print(f"✓ [Modul 2] Model Terbaik: {'Random Forest' if metrics_rf['accuracy'] > metrics_lr['accuracy'] else 'Logistic Regression'}")
print(f"✓ [Modul 2] Akurasi Terbaik: {max(metrics_rf['accuracy'], metrics_lr['accuracy']):.4f}")
print("✓ [Modul 3] Berkas Serialisasi Disimpan di: results/models/")
print("✓ [Modul 3] Berkas Pembersihan Data Disimpan di: data/processed/dataset_clean.csv")
print("="*60)

print("\n📚 Langkah Selanjutnya untuk Mahasiswa:")
print("1. Buka dan pelajari lembar kerja interaktif di folder 'notebooks/' (Modul 2).")
print("2. Jalankan pengujian deployment menggunakan: python test_deployment.py")
print("3. Luncurkan aplikasi web dashboard menggunakan: streamlit run app.py (Modul 3).")
print("\nSelamat belajar!")
