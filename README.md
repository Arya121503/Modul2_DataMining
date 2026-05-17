# 📊 Modul 2 - Pemodelan Data (Data Modeling)

Repository ini berisi alur lengkap praktikum pemodelan data untuk Data Mining.

## 📋 Daftar Isi
- [Instalasi](#instalasi)
- [Struktur Project](#struktur-project)
- [Library yang Digunakan](#library-yang-digunakan)
- [Alur Kerja](#alur-kerja)
- [Panduan Praktikum](#panduan-praktikum)
- [Tips & Troubleshooting](#tips--troubleshooting)

## 🔧 Instalasi

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip (package installer)

### Step-by-Step Installation

```bash
# 1. Clone atau download project ini
cd modul2_datamining

# 2. Buat virtual environment
python -m venv venv

# 3. Aktivasi virtual environment
# Di Windows:
venv\Scripts\activate
# Di Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Buka Jupyter Notebook
jupyter notebook
```

## 📁 Struktur Project

```
modul2_datamining/
├── data/                              # Folder data
│   ├── raw/                          # Data original (jangan dimodifikasi)
│   │   └── dataset.csv               # Dataset mentah dummy untuk praktikum
│   └── processed/                    # Data yang sudah diproses (dibuat saat notebook dijalankan)
│       └── dataset_clean.csv         # Dataset siap modeling
│
├── src/                              # Source code modules
│   ├── __init__.py                  # Python package initializer
│   ├── data_loader.py               # Modul loading data
│   ├── preprocessing.py             # Modul data cleaning
│   ├── feature_engineering.py       # Modul feature engineering
│   ├── model.py                     # Modul building model
│   └── evaluation.py                # Modul evaluasi model
│
├── notebooks/                       # Jupyter Notebooks
│   ├── 1_eda.ipynb                 # Exploratory Data Analysis
│   ├── 2_preprocessing.ipynb       # Data Preprocessing
│   └── 3_modeling.ipynb            # Model Building & Evaluation
│
├── results/                         # Output dan hasil analisis
│   ├── models/                     # Trained models (pickle files)
│   └── reports/                    # Laporan, visualisasi, hasil
│
├── requirements.txt                # List semua library
├── ALUR_PRAKTIKUM.md              # Panduan alur praktikum
└── README.md                       # File ini
```

## 📦 Dataset yang Disarankan

Jika mahasiswa ingin mencoba dataset lain, pilih data yang:

- Memiliki kolom numerik dan kategorikal.
- Memiliki target klasifikasi biner agar notebook modeling tetap sederhana.
- Ukurannya tidak terlalu besar supaya cepat diproses di kelas.

Rekomendasi dataset yang cocok:

| Dataset | Kelebihan | Catatan |
|---------|-----------|---------|
| Titanic | Mudah dipahami, ada numerik dan kategorikal | Cocok untuk klasifikasi biner |
| Breast Cancer Wisconsin | Bersih, cepat, cocok untuk demo model | Hampir semua kolom numerik |
| Adult Income | Realistis, kaya fitur kategorikal | Perlu encoding yang baik |
| Bank Marketing | Cocok untuk klasifikasi dan evaluasi | Ukuran sedang, bagus untuk latihan |

Jika memakai dataset sendiri, simpan file sebagai `data/raw/dataset.csv` atau sesuaikan nama file di notebook.

## 📚 Library yang Digunakan

### Data Science & Analytics
| Library | Versi | Fungsi |
|---------|-------|--------|
| **pandas** | 2.0.3 | Data manipulation & analysis |
| **numpy** | 1.24.3 | Numerical computing |
| **scipy** | 1.11.2 | Scientific computing |

### Visualization
| Library | Versi | Fungsi |
|---------|-------|--------|
| **matplotlib** | 3.7.2 | Basic plotting |
| **seaborn** | 0.12.2 | Statistical data visualization |
| **plotly** | 5.16.1 | Interactive plots (optional) |

### Machine Learning
| Library | Versi | Fungsi |
|---------|-------|--------|
| **scikit-learn** | 1.3.1 | ML algorithms, preprocessing |

### Development Tools
| Library | Versi | Fungsi |
|---------|-------|--------|
| **jupyter** | 1.0.0 | Interactive notebooks |
| **ipython** | 8.15.0 | Enhanced Python shell |

## 🔄 Alur Kerja

### 1️⃣ Step 1: Data Loading
**File:** `src/data_loader.py`

```python
from src.data_loader import DataLoader

loader = DataLoader("data/raw")
df = loader.load_csv("dataset.csv")
loader.get_data_info(df)
```

**Output:**
- Menampilkan shape data
- Tipe data setiap kolom
- Missing values per kolom
- Statistik deskriptif

---

### 2️⃣ Step 2: Exploratory Data Analysis (EDA)
**File:** `notebooks/1_eda.ipynb`

**Aktivitas:**
- 📊 Visualisasi distribusi data
- 🔗 Analisis korelasi
- 📈 Statistik deskriptif
- 🔍 Identifikasi patterns

---

### 3️⃣ Step 3: Data Preprocessing
**File:** `src/preprocessing.py`

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Handle missing values
df = preprocessor.handle_missing_values(df, strategy='mean')

# Handle outliers
df = preprocessor.handle_outliers(df, numeric_cols, method='iqr')

# Encode categorical variables
df = preprocessor.encode_categorical(df, categorical_cols)

# Scale/Normalize
df = preprocessor.normalize_scale(df, numeric_cols, method='standard')
```

---

### 4️⃣ Step 4: Feature Engineering
**File:** `src/feature_engineering.py`

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Create interaction features
df = engineer.create_interaction_features(df, 'col1', 'col2')

# Polynomial features
df = engineer.create_polynomial_features(df, 'col1', degree=2)

# Feature selection
best_features = engineer.select_best_features(X, y, k=10)

# PCA
X_pca = engineer.apply_pca(X, n_components=0.95)
```

---

### 5️⃣ Step 5: Model Building
**File:** `src/model.py`

```python
from src.model import ModelBuilder

builder = ModelBuilder()

# Split data
X_train, X_test, y_train, y_test = builder.split_data(X, y)

# Train model
model = builder.train_random_forest(X_train, y_train)

# Hyperparameter tuning
best_model = builder.hyperparameter_tuning(
    X_train, y_train, 
    model_name='random_forest',
    params={...}
)

# Save model
builder.save_model(best_model, 'results/models/model.pkl')
```

---

### 6️⃣ Step 6: Model Evaluation
**File:** `src/evaluation.py`

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Get predictions
y_pred = model.predict(X_test)

# Evaluate
metrics = evaluator.evaluate_classification(y_test, y_pred)

# Visualize
evaluator.plot_confusion_matrix(y_test, y_pred)
evaluator.plot_roc_curve(y_test, y_pred_proba)
evaluator.plot_feature_importance(feature_imp_df)

# Compare models
results = evaluator.compare_models(
    {'Model1': model1, 'Model2': model2},
    X_test, y_test
)
```

---

### 7️⃣ Step 7: Dokumentasi & Reporting
- Buat kesimpulan dari hasil modeling
- Visualisasi hasil akhir
- Rekomendasi untuk deployment

---

## 💡 Tips & Best Practices

### ✅ Do's
- ✓ Selalu backup data original
- ✓ Gunakan version control (git)
- ✓ Document setiap step
- ✓ Test dengan cross-validation
- ✓ Visualize sebelum modeling
- ✓ Monitor performance metrics

### ❌ Don'ts
- ✗ Jangan modifikasi data original
- ✗ Jangan skip EDA
- ✗ Jangan overfit (gunakan validation)
- ✗ Jangan ignore class imbalance
- ✗ Jangan mix train dan test data

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pandas'"
```bash
# Solution:
pip install pandas
# atau install semua requirements:
pip install -r requirements.txt
```

### Error: "No such file or directory: 'data/raw/dataset.csv'"
```
• Pastikan file dataset ada di folder data/raw/
• Check path relatif dari working directory
• Gunakan absolute path jika perlu
```

### Memory Error pada large dataset
```python
# Solution: Baca data dalam chunks
df = pd.read_csv('file.csv', chunksize=10000)
for chunk in df:
    # Process chunk
    pass
```

### Slow performance saat training
```python
# Solution: 
# 1. Gunakan n_jobs=-1 untuk parallel processing
# 2. Reduce feature size
# 3. Gunakan sample data untuk testing
```

---

## 📞 Support & Resources

### Documentation
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Jupyter Documentation](https://jupyter.org/)

### Useful Tutorials
- Data cleaning with Pandas
- Machine learning with Scikit-learn
- Data visualization with Matplotlib & Seaborn
- Feature engineering techniques

---

## 📝 Checklist Praktikum

Gunakan checklist ini untuk memastikan semua step selesai:

- [ ] Data loaded dan diinspeksi
- [ ] EDA completed dengan insights
- [ ] Preprocessing selesai (missing values, outliers, encoding)
- [ ] Feature engineering applied
- [ ] Data split menjadi train-test
- [ ] Models trained dan evaluated
- [ ] Best model selected
- [ ] Model saved
- [ ] Hasil didokumentasikan

---

## 📄 License

Modul praktikum untuk keperluan pendidikan.

---

**Happy Learning! 🚀**

Untuk pertanyaan atau issues, silakan discuss atau buat pull request.
