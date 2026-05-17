# 🚀 START HERE - Panduan Awal Praktikum

Selamat datang di **Modul 2 - Pemodelan Data (Data Modeling)**! 

Ini adalah project praktikum lengkap untuk belajar data mining & machine learning.

---

## ⚡ 5 Menit Quick Start

### Step 1: Setup (2 menit)
```bash
# Buka terminal di folder modul2_datamining

# 1. Buat virtual environment
python -m venv venv

# 2. Aktivasi (Windows)
venv\Scripts\activate
# atau (Linux/Mac)
source venv/bin/activate

# 3. Install libraries
pip install -r requirements.txt
```

### Step 2: Jalankan Demo (1 menit)
```bash
# Jalankan contoh lengkap dengan sample data
python quick_start.py
```

### Step 3: Explore (2 menit)
```bash
# Buka Jupyter
jupyter notebook

# Buka notebooks/1_eda.ipynb untuk explore lebih lanjut
```

---

## 📚 Dokumentasi Lengkap

Baca file-file ini sesuai urutan:

### 1️⃣ **[README.md](README.md)** ← BACA INI DULU (15 menit)
Dokumentasi lengkap yang mencakup:
- Instalasi lengkap
- Penjelasan setiap library
- 7 steps alur kerja dengan contoh code
- Tips & troubleshooting

### 2️⃣ **[ALUR_PRAKTIKUM.md](ALUR_PRAKTIKUM.md)** (5 menit)
Ringkasan singkat:
- Struktur project
- Library yang digunakan
- Alur kerja 7 steps

### 3️⃣ **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** (10 menit)
Detail setiap file:
- Penjelasan fungsi setiap module
- Contoh penggunaan setiap class
- Method-method yang tersedia

### 4️⃣ **[CHECKLIST.md](CHECKLIST.md)** (5 menit)
Checklist setup:
- File yang telah dibuat
- Pre-praktikum checklist
- Learning path

---

## 🎯 Alur 7 Steps Praktikum

Praktikum ini mengikuti 7 steps berikut:

```
1. DATA LOADING           → Load dan explore data
2. EDA (Analisis)         → Pahami data dengan visualisasi
3. PREPROCESSING          → Clean data (missing, outliers, encoding)
4. FEATURE ENGINEERING    → Buat & pilih features
5. MODEL BUILDING         → Train berbagai models
6. MODEL EVALUATION       → Bandingkan & evaluate
7. DOCUMENTATION          → Simpan & document hasil
```

Setiap step memiliki:
- 📄 Python module siap pakai di folder `src/`
- 📓 Jupyter notebook template (di folder `notebooks/`)
- 💻 Contoh code di `quick_start.py`

---

## 📁 File Structure

```
modul2_datamining/
├── 📄 README.md                ← Baca ini untuk dokumentasi lengkap
├── 📄 ALUR_PRAKTIKUM.md       ← Ringkasan alur
├── 📄 PROJECT_STRUCTURE.md    ← Detail struktur project
├── 📄 CHECKLIST.md            ← Pre-praktikum checklist
├── 📄 START_HERE.md           ← File ini!
├── 📄 requirements.txt         ← List library
├── 📄 quick_start.py           ← Contoh jalan (python quick_start.py)
│
├── 📁 src/                     ← Python modules (siap pakai!)
│   ├── data_loader.py          ← Step 1: Load data
│   ├── preprocessing.py        ← Step 3: Clean data
│   ├── feature_engineering.py  ← Step 4: Feature engineering
│   ├── model.py                ← Step 5: Train models
│   └── evaluation.py           ← Step 6: Evaluate models
│
├── 📁 data/
│   ├── raw/                    ← Letakkan dataset.csv di sini!
│   └── processed/              ← Hasil preprocessing
│
├── 📁 notebooks/               ← Jupyter notebooks
│   ├── 1_eda.ipynb            ← Exploratory analysis
│   ├── 2_preprocessing.ipynb   ← Data cleaning
│   └── 3_modeling.ipynb       ← Model building
│
└── 📁 results/
    ├── models/                 ← Trained models disimpan di sini
    └── reports/                ← Analysis reports & plots
```

---

## 💾 Siapkan Dataset Anda

1. **Download atau siapkan dataset CSV Anda**
2. **Letakkan di folder:** `data/raw/dataset.csv`
3. **Ubah nama file di code jika berbeda** (search & replace `dataset.csv`)

Format yang diterima:
- ✅ CSV dengan headers (kolom names)
- ✅ Numeric & categorical columns
- ✅ Boleh ada missing values (akan dihandle)

---

## 🚀 3 Cara Menggunakan Project Ini

### **Cara 1: Quick Demo (Paling Cepat)**
Cocok jika Anda ingin lihat semua 7 steps dalam 1 script:

```bash
python quick_start.py
```

Output: Menampilkan hasil semua 7 steps menggunakan sample data.

---

### **Cara 2: Jupyter Notebook (Paling Interaktif)**
Cocok untuk explorasi & learning:

```bash
jupyter notebook
# Buat atau buka notebooks di folder notebooks/
```

Keuntungan:
- ✅ Lihat output setiap cell
- ✅ Visualisasi langsung
- ✅ Easy to modify & experiment

---

### **Cara 3: Custom Python Script (Paling Fleksibel)**
Cocok jika ingin custom alur:

```python
# my_analysis.py
from src import DataLoader, DataPreprocessor, ModelBuilder, ModelEvaluator

# Buat custom pipeline sesuai kebutuhan
loader = DataLoader("data/raw")
df = loader.load_csv("dataset.csv")

# ... lanjutkan custom code
```

---

## 📚 Python Modules (Siap Pakai!)

Project ini sudah memiliki 5 modules siap pakai:

### **1. DataLoader** (data_loader.py)
```python
from src.data_loader import DataLoader

loader = DataLoader("data/raw")
df = loader.load_csv("dataset.csv")
loader.get_data_info(df)
```
Functions: `load_csv()`, `get_data_info()`, `check_duplicates()`

### **2. DataPreprocessor** (preprocessing.py)
```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.handle_missing_values(df, strategy='mean')
df = preprocessor.encode_categorical(df, ['col1', 'col2'])
df = preprocessor.normalize_scale(df, numeric_cols)
```
Functions: `handle_missing_values()`, `handle_outliers()`, `encode_categorical()`, `normalize_scale()`

### **3. FeatureEngineer** (feature_engineering.py)
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
best_features = engineer.select_best_features(X, y, k=10)
X_pca = engineer.apply_pca(X, n_components=0.95)
```
Functions: `select_best_features()`, `apply_pca()`, `create_polynomial_features()`, dll

### **4. ModelBuilder** (model.py)
```python
from src.model import ModelBuilder

builder = ModelBuilder()
X_train, X_test, y_train, y_test = builder.split_data(X, y)
model = builder.train_random_forest(X_train, y_train)
builder.save_model(model, 'results/models/model.pkl')
```
Functions: `train_random_forest()`, `hyperparameter_tuning()`, `cross_validate()`, dll

### **5. ModelEvaluator** (evaluation.py)
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_classification(y_test, y_pred)
evaluator.plot_confusion_matrix(y_test, y_pred)
results = evaluator.compare_models(models_dict, X_test, y_test)
```
Functions: `evaluate_classification()`, `plot_confusion_matrix()`, `compare_models()`, dll

---

## 📊 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **pandas** | Manipulasi data |
| **numpy** | Numerical computing |
| **scikit-learn** | Machine learning |
| **matplotlib** | Plotting |
| **seaborn** | Statistical visualization |
| **jupyter** | Interactive notebooks |

Total 13 libraries di `requirements.txt`

---

## ✅ Checklist Setup

- [ ] Install Python 3.8+
- [ ] Download/clone project
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate venv: `venv\Scripts\activate`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Verify: `python quick_start.py`
- [ ] Prepare dataset di `data/raw/`
- [ ] Baca README.md

---

## 🎓 Learning Path (6 Minggu)

```
Minggu 1: Setup & Understanding
  • Setup environment
  • Baca dokumentasi
  • Jalankan quick_start.py

Minggu 2: Data Loading & EDA
  • Pahami data Anda
  • Create notebook 1_eda.ipynb
  • Explore dengan visualisasi

Minggu 3: Preprocessing
  • Clean data
  • Handle missing values & outliers
  • Create notebook 2_preprocessing.ipynb

Minggu 4: Feature Engineering & Modeling
  • Create features
  • Train models
  • Create notebook 3_modeling.ipynb

Minggu 5: Evaluation
  • Evaluate models
  • Hyperparameter tuning
  • Compare performance

Minggu 6: Final & Presentation
  • Select best model
  • Document findings
  • Present hasil
```

---

## 🆘 Troubleshooting

### Q: "Module not found" error?
**A:** 
```bash
# Make sure requirements installed:
pip install -r requirements.txt
```

### Q: Where to put my dataset?
**A:** 
```
modul2_datamining/data/raw/dataset.csv
```

### Q: How to modify quick_start.py?
**A:** 
Buka file `quick_start.py`, cari & replace nama file dataset (currently: `dataset.csv`)

### Q: How to use my own modules?
**A:** 
Buat file Python di folder `src/` dan import:
```python
from src.my_module import MyClass
```

### Q: Jupyter notebook kernel error?
**A:** 
```bash
# Restart kernel di Jupyter UI
# atau di terminal:
jupyter kernelspec list
```

---

## 📞 Quick Reference

### File yang paling penting:
1. **[README.md](README.md)** - Dokumentasi lengkap
2. **[requirements.txt](requirements.txt)** - Install dependencies
3. **[quick_start.py](quick_start.py)** - Contoh lengkap
4. **[src/](src/)** - Module siap pakai

### Commands yang sering digunakan:
```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run
python quick_start.py
jupyter notebook

# Check
python -c "import pandas; print('OK')"
```

---

## 🎯 Next Action (Pilih 1)

### ✅ Jika Anda ingin CEPAT:
1. Run: `python quick_start.py`
2. Baca: [README.md](README.md)

### ✅ Jika Anda ingin MENGERTI DETAIL:
1. Baca: [README.md](README.md) (15 menit)
2. Baca: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (10 menit)
3. Run: `python quick_start.py`
4. Explore: Jupyter notebook

### ✅ Jika Anda ingin LANGSUNG PRAKTEK:
1. Siapkan dataset di `data/raw/`
2. Edit: `quick_start.py` (ubah nama dataset)
3. Run: `python quick_start.py`
4. Modify & experiment

---

## 🎉 Selamat Dimulai!

Project ini dirancang untuk membuat pembelajaran data modeling menjadi mudah & terstruktur.

**Total waktu setup: 15 menit**
**Total waktu first run: 5 menit**

Berikutnya, baca [README.md](README.md) untuk dokumentasi lengkap.

**Happy Learning! 🚀**

---

### File dibuat: May 17, 2026
### Status: ✅ Ready to use
### Version: 1.0.0
