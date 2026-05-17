# Alur Praktikum Modul 2: Pemodelan Data

## 📋 Struktur Project

```
modul2_datamining/
│
├── data/
│   ├── raw/                      # Data asli (tidak dimodifikasi)
│   │   └── dataset.csv           # Dataset mentah
│   └── processed/                # Data yang sudah diproses
│       └── dataset_clean.csv
│
├── notebooks/                    # Jupyter Notebooks untuk eksplorasi
│   ├── 1_eda.ipynb              # Exploratory Data Analysis
│   ├── 2_data_preprocessing.ipynb # Persiapan data
│   └── 3_modeling.ipynb         # Pemodelan dan evaluasi
│
├── src/                         # Source code
│   ├── data_loader.py           # Loading dan membaca data
│   ├── preprocessing.py         # Data cleaning & preparation
│   ├── feature_engineering.py   # Feature creation
│   ├── model.py                 # Model building
│   └── evaluation.py            # Model evaluation
│
├── results/                     # Hasil analisis
│   ├── models/                  # Model yang sudah dilatih
│   └── reports/                 # Laporan dan visualisasi
│
├── requirements.txt             # Daftar library yang digunakan
└── README.md                    # Dokumentasi project
```

---

## 📚 Library yang Digunakan

### 1. **Data Manipulation & Analysis**
- **pandas** - Manipulasi dan analisis data
- **numpy** - Operasi numerik

### 2. **Visualization**
- **matplotlib** - Plotting dasar
- **seaborn** - Visualisasi statistik
- **plotly** - Visualisasi interaktif (opsional)

### 3. **Machine Learning**
- **scikit-learn** - Model machine learning, preprocessing, evaluation

### 4. **Data Processing**
- **scipy** - Operasi ilmiah dan statistik

### 5. **Development Tools**
- **jupyter** - Notebook untuk eksplorasi
- **ipython** - Interactive shell

---

## 🔄 Alur Kerja Praktikum

### **Step 1: Persiapan Data (Data Preparation)**
- Membaca dataset
- Mengenal struktur data
- Identifikasi missing values
- Identifikasi outliers

### **Step 2: Exploratory Data Analysis (EDA)**
- Statistik deskriptif
- Distribusi data
- Korelasi antar variabel
- Visualisasi data

### **Step 3: Data Preprocessing**
- Handle missing values
- Handle outliers
- Normalisasi/Standardisasi
- Encoding kategori

### **Step 4: Feature Engineering**
- Membuat fitur baru
- Feature selection
- Dimensionality reduction (jika diperlukan)

### **Step 5: Pemodelan (Modeling)**
- Split data (train-test)
- Membangun model
- Hyperparameter tuning
- Cross-validation

### **Step 6: Evaluasi Model**
- Metrik evaluasi
- Confusion matrix
- ROC-AUC curve
- Classification report

### **Step 7: Interpretasi & Dokumentasi**
- Analisis hasil
- Kesimpulan
- Rekomendasi

---

## 📦 Installation

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Navigate ke folder project
cd modul2_datamining

# 2. Buka Jupyter Notebook
jupyter notebook

# 3. Buka file 1_eda.ipynb untuk memulai
```

---

## 📝 Tips Praktikum

1. ✅ Selalu backup data original di folder `data/raw/`
2. ✅ Dokumentasikan setiap step dengan comment yang jelas
3. ✅ Visualisasi data sebelum membuat model
4. ✅ Gunakan version control (git) untuk tracking perubahan
5. ✅ Test model dengan data yang berbeda
6. ✅ Simpan model yang baik untuk prediksi masa depan

