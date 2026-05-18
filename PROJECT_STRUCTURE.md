# 📋 Ringkasan File & Struktur Project

## 📁 Struktur Lengkap yang Telah Dibuat

```
modul2_datamining/
│
├── 📄 README.md                          ← Dokumentasi utama (START HERE!)
├── 📄 ALUR_PRAKTIKUM.md                 ← Panduan alur praktikum
├── 📄 SRC_PENJELASAN.md                 ← Penjelasan sederhana isi src
├── 📄 requirements.txt                   ← Daftar library yang diperlukan
├── 📄 quick_start.py                     ← Script contoh cepat
├── 📄 PROJECT_STRUCTURE.md              ← File ini
│
├── 📁 data/                             ← Folder data
│   ├── raw/                             ← Data original (jangan dimodifikasi)
│   │   └── [Letakkan dataset.csv di sini]
│   └── processed/                       ← Data yang sudah diproses
│       └── [Hasil preprocessing akan disimpan di sini]
│
├── 📁 src/                              ← Source code modules
│   ├── __init__.py                      ← Python package initializer
│   ├── data_loader.py                   ← Modul 1: Loading data
│   ├── preprocessing.py                 ← Modul 2: Data cleaning
│   ├── feature_engineering.py           ← Modul 3: Feature engineering
│   ├── model.py                         ← Modul 4: Model building
│   ├── evaluation.py                    ← Modul 5: Model evaluation
│   └── clustering.py                    ← Modul 6: Clustering unsupervised
│
├── 📁 notebooks/                        ← Jupyter Notebooks (kosong - dibuat saat praktikum)
│   ├── 1_eda.ipynb                      ← Exploratory Data Analysis
│   ├── 2_preprocessing.ipynb            ← Data preprocessing
│   ├── 3_modeling.ipynb                 ← Model building & evaluation
│   └── 4_clustering.ipynb               ← Clustering & unsupervised learning
│
└── 📁 results/                          ← Hasil & output
    ├── models/                          ← Trained models (.pkl files)
    └── reports/                         ← Laporan dan visualisasi
```

---

## 📚 Penjelasan Setiap File

### **1. README.md** 📖
File dokumentasi utama yang berisi:
- Daftar library dan versi
- Panduan instalasi lengkap
- Penjelasan alur kerja 7 steps
- Tips & troubleshooting
- Quick start guide

**👉 Mulai dari sini!**

---

### **2. ALUR_PRAKTIKUM.md** 🎯
Panduan singkat alur praktikum yang mencakup:
- Struktur project
- Library yang digunakan
- 7 step alur kerja
- Installation instructions
- Tips praktikum

---

### **3. requirements.txt** 📦
Daftar semua library Python yang diperlukan dengan versi spesifiknya:

```
pandas==2.0.3          # Data manipulation
numpy==1.24.3          # Numerical computing
scipy==1.11.2          # Scientific computing
matplotlib==3.7.2      # Visualization
seaborn==0.12.2        # Statistical plots
scikit-learn==1.3.1    # Machine learning
jupyter==1.0.0         # Notebooks
```

**Cara menggunakan:**
```bash
pip install -r requirements.txt
```

---

### **4. quick_start.py** 🚀
Script Python standalone yang menjalankan semua 7 steps praktikum:
1. Data Loading
2. EDA
3. Preprocessing
4. Feature Engineering
5. Model Building
6. Model Evaluation
7. Save Results

**Cara menjalankan:**
```bash
python quick_start.py
```

---

## 🔧 Source Code Modules (src/)

### **src/__init__.py**
Membuat folder `src` menjadi Python package yang bisa diimport.

```python
from src import DataLoader, DataPreprocessor, FeatureEngineer, ModelBuilder, ModelEvaluator, ClusterAnalyzer
```

---

### **src/data_loader.py** 📥
**Fungsi:** Loading dan membaca data

**Kelas:** `DataLoader`

**Metode penting:**
- `load_csv(filename)` - Baca file CSV
- `get_data_info(df)` - Tampilkan info data
- `check_duplicates(df)` - Cek duplikat

**Contoh penggunaan:**
```python
from src.data_loader import DataLoader

loader = DataLoader("data/raw")
df = loader.load_csv("dataset.csv")
loader.get_data_info(df)
```

---

### **src/preprocessing.py** 🧹
**Fungsi:** Data cleaning dan preprocessing

**Kelas:** `DataPreprocessor`

**Metode penting:**
- `handle_missing_values()` - Handle missing data (mean, median, drop, forward_fill)
- `handle_outliers()` - Handle outliers (IQR, zscore)
- `encode_categorical()` - Encode kategori (label encoding, one-hot)
- `normalize_scale()` - Normalisasi data (standard, minmax)

**Contoh penggunaan:**
```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.handle_missing_values(df, strategy='mean')
df = preprocessor.encode_categorical(df, ['category_col'])
df = preprocessor.normalize_scale(df, numeric_cols, method='standard')
```

---

### **src/feature_engineering.py** ⚙️
**Fungsi:** Feature engineering dan feature selection

**Kelas:** `FeatureEngineer`

**Metode penting:**
- `create_interaction_features()` - Buat fitur interaksi
- `create_polynomial_features()` - Buat fitur polynomial
- `create_binning_features()` - Buat fitur binning
- `select_best_features()` - Feature selection (SelectKBest)
- `apply_pca()` - Dimensionality reduction
- `get_feature_importance()` - Get feature importance

**Contoh penggunaan:**
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
df = engineer.create_interaction_features(df, 'col1', 'col2')
best_features = engineer.select_best_features(X, y, k=10)
```

---

### **src/model.py** 🤖
**Fungsi:** Building dan training model machine learning

**Kelas:** `ModelBuilder`

**Metode penting:**
- `split_data()` - Split train-test
- `train_logistic_regression()` - Logistic Regression
- `train_random_forest()` - Random Forest
- `train_gradient_boosting()` - Gradient Boosting
- `train_svm()` - Support Vector Machine
- `cross_validate()` - Cross-validation
- `hyperparameter_tuning()` - GridSearchCV
- `save_model()` - Simpan model
- `load_model()` - Load model

**Contoh penggunaan:**
```python
from src.model import ModelBuilder

builder = ModelBuilder()
X_train, X_test, y_train, y_test = builder.split_data(X, y)
model = builder.train_random_forest(X_train, y_train)
best_model = builder.hyperparameter_tuning(X_train, y_train, model_name='random_forest')
builder.save_model(best_model, 'results/models/best_model.pkl')
```

---

### **src/evaluation.py** 📊
**Fungsi:** Evaluasi model performance

**Kelas:** `ModelEvaluator`

**Metode penting:**
- `evaluate_classification()` - Metrics klasifikasi (accuracy, precision, recall, f1, roc_auc)
- `evaluate_regression()` - Metrics regresi (mse, rmse, mae, r2)
- `get_classification_report()` - Detailed report
- `get_confusion_matrix()` - Confusion matrix
- `plot_confusion_matrix()` - Plot confusion matrix
- `plot_roc_curve()` - Plot ROC curve
- `plot_feature_importance()` - Plot feature importance
- `plot_predictions_vs_actual()` - Plot predictions vs actual
- `compare_models()` - Bandingkan multiple models

**Contoh penggunaan:**
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
y_pred = model.predict(X_test)
metrics = evaluator.evaluate_classification(y_test, y_pred)
evaluator.plot_confusion_matrix(y_test, y_pred)
comparison = evaluator.compare_models({'Model1': m1, 'Model2': m2}, X_test, y_test)
```

---

### **src/clustering.py** 🧭
**Fungsi:** Clustering dan evaluasi unsupervised learning

**Kelas:** `ClusterAnalyzer`

**Metode penting:**
- `prepare_features()` - pilih dan scale fitur numerik
- `fit_kmeans()` - latih KMeans
- `evaluate_clustering()` - hitung silhouette score dan ARI
- `find_best_k()` - coba beberapa nilai cluster
- `reduce_to_2d()` - reduksi dimensi untuk visualisasi

**Contoh penggunaan: **
```python
from src.clustering import ClusterAnalyzer

clusterer = ClusterAnalyzer()
X_scaled = clusterer.prepare_features(df)
labels = clusterer.fit_kmeans(X_scaled, n_clusters=3)
metrics = clusterer.evaluate_clustering(X_scaled, labels, y_true=df['purchased'])
```

---

## 🚀 Cara Memulai

### **Opsi 1: Quick Start (Cepat)**
```bash
cd modul2_datamining
python quick_start.py
```

### **Opsi 2: Jupyter Notebook (Interaktif)**
```bash
cd modul2_datamining
jupyter notebook
# Buka notebooks/1_eda.ipynb, 2_preprocessing.ipynb, 3_modeling.ipynb, 4_clustering.ipynb
```

### **Opsi 3: Python Script (Custom)**
```python
# Buat file baru: my_project.py
from src import DataLoader, DataPreprocessor, ModelBuilder, ModelEvaluator

# Gunakan modules sesuai kebutuhan
```

---

## 📖 Alur 7 Steps Praktikum

```
Step 1: DATA LOADING
        ↓
Step 2: EXPLORATORY DATA ANALYSIS (EDA)
        ↓
Step 3: DATA PREPROCESSING
        ↓
Step 4: FEATURE ENGINEERING
        ↓
Step 5: MODEL BUILDING
        ↓
Step 6: MODEL EVALUATION
        ↓
Step 7: DOCUMENTATION & REPORTING
```

---

## ✅ Checklist Setup

- [ ] Install Python 3.8+
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate venv: `venv\Scripts\activate`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Verify installation: `python -c "import pandas; import sklearn"`
- [ ] Place dataset in `data/raw/` folder
- [ ] Run quick_start: `python quick_start.py`
- [ ] Open Jupyter: `jupyter notebook`

---

## 💡 Tips

1. **Selalu backup** data original di `data/raw/`
2. **Dokumentasi** setiap step dengan comment
3. **Test dengan data kecil** dulu sebelum full data
4. **Visualisasi** data sebelum modeling
5. **Cross-validate** untuk validasi model
6. **Save model** yang terbaik untuk predictions

---

## 🆘 Troubleshooting

### Module tidak ditemukan?
```bash
# Pastikan venv activated dan requirements installed
pip install -r requirements.txt
```

### File dataset tidak ditemukan?
```
• Letakkan file di: data/raw/dataset.csv
• Check nama file sesuai path yang digunakan
```

### Import error di Jupyter?
```
• Restart kernel
• Pastikan bekerja di working directory yang benar
```

---

## 📞 Resources

- [Pandas Docs](https://pandas.pydata.org/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Matplotlib Docs](https://matplotlib.org/)
- [Jupyter Docs](https://jupyter.org/)

---

**Selamat Praktikum! 🎉**

Start dengan README.md untuk dokumentasi lengkap.
