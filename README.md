# 📚 Panduan Praktikum: Data Mining & Model Deployment

Selamat datang di repository praktikum Data Mining. Repository ini dirancang khusus untuk membantu mahasiswa memahami alur pemodelan data (Modul 2) hingga tahap deployment model menjadi aplikasi web interaktif (Modul 3).

---

## 🛠️ Langkah Awal: Setup Environment
Sebelum memulai salah satu modul, pastikan Anda telah menyiapkan environment Python terlebih dahulu:

1. **Buka Terminal/Command Prompt** di folder project `modul2_datamining`.
2. **Buat Virtual Environment**:
   ```bash
   python -m venv env
   ```
3. **Aktifkan Virtual Environment**:
   * **Windows**:
     ```powershell
     env\Scripts\activate
     ```
   * **Linux / macOS**:
     ```bash
     source env/bin/activate
     ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Struktur Folder Proyek
Berikut adalah pembagian file dan folder berdasarkan materi praktikum:

```text
modul2_datamining/
│
├── 📄 requirements.txt               # Dependencies/library Python yang diperlukan
├── 📄 README.md                      # Dokumentasi utama (file ini)
├── 📄 SRC_PENJELASAN.md              # Penjelasan detail modul di folder src/
│
├── 📁 data/                          # Manajemen data
│   ├── raw/                          # Dataset mentah (original)
│   │   └── customer_purchase_data.csv
│   └── processed/                    # Dataset hasil pembersihan (clean)
│       └── dataset_clean.csv
│
├── 📁 src/                           # Modul kode Python yang digunakan bersama
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluation.py
│
├── 📁 notebooks/                     # Jupyter Notebook untuk lembar kerja mahasiswa
│   ├── 1_eda.ipynb                  # Modul 2: Eksplorasi Data
│   ├── 2_preprocessing.ipynb        # Modul 2: Pembersihan Data
│   ├── 3_modeling.ipynb             # Modul 2: Pemodelan & Evaluasi
│   └── 4_clustering.ipynb           # Topik Tambahan: Unsupervised Learning (Clustering)
│
├── 📁 results/                       # Penyimpanan model dan laporan
│   └── models/                       # File model ter-serialisasi (.pkl) untuk Modul 3
│
├── 📄 quick_start.py                 # Script otomasi alur Modul 2 & 3
├── 📄 app.py                         # Modul 3: Dashboard Web Deployment Streamlit
└── 📄 test_deployment.py             # Modul 3: Script verifikasi fungsionalitas model
```

---

## 🧪 MODUL 2: PEMODELAN DATA (MACHINE LEARNING PIPELINE)
Modul ini berfokus pada alur pembuatan model machine learning untuk klasifikasi biner, mulai dari membaca data hingga melakukan evaluasi performa.

### A. Alur Kerja Praktikum Modul 2
Alur kerja pemodelan diimplementasikan secara modular di dalam folder `src/`:
1. **Data Loading** (`src/data_loader.py`): Memuat dataset mentah dari format CSV.
2. **Exploratory Data Analysis (EDA)** (`notebooks/1_eda.ipynb`): Memahami karakteristik data melalui visualisasi dan statistik deskriptif.
3. **Data Preprocessing** (`src/preprocessing.py`): Menangani missing values, outliers, standardisasi skala, dan encoding kolom kategorikal.
4. **Feature Engineering** (`src/feature_engineering.py`): Membuat fitur baru, melakukan seleksi fitur terbaik, atau reduksi dimensi (PCA).
5. **Model Building** (`src/model.py`): Memisahkan data (train-test split), melatih algoritma (Logistic Regression, Random Forest, SVM), serta optimasi parameter menggunakan GridSearchCV.
6. **Model Evaluation** (`src/evaluation.py`): Mengukur kinerja model menggunakan metrik akurasi, presisi, recall, F1-score, Confusion Matrix, dan kurva ROC.

### B. Cara Menjalankan Modul 2
* **Opsi Interaktif (Jupyter Notebook)**:
  Jalankan perintah berikut di terminal:
  ```bash
  jupyter notebook
  ```
  Buka dan kerjakan lembar kerja mahasiswa secara berurutan di dalam folder `notebooks/` mulai dari `1_eda.ipynb` hingga `3_modeling.ipynb`.
  
* **Opsi Cepat (Otomatis)**:
  Jalankan script python berikut untuk mengeksekusi seluruh pipeline dari data loading hingga model training secara otomatis:
  ```bash
  python quick_start.py
  ```

---

## 🚀 MODUL 3: MODEL DEPLOYMENT (DASHBOARD WEB)
Modul ini berfokus pada bagaimana model yang telah dilatih pada Modul 2 dapat disimpan dan diintegrasikan ke dalam antarmuka web interaktif menggunakan framework **Streamlit**.

### A. Cara Kerja Deployment
1. **Serialisasi Model**:
   Saat menjalankan `quick_start.py`, model terbaik dan objek preprocessing diserialisasi (disimpan) menggunakan modul `pickle` ke dalam folder `results/models/` sebagai berkas berikut:
   * `best_model.pkl`: Model terbaik yang telah dilatih.
   * `preprocessor.pkl`: Objek encoder kolom kategorikal.
   * `scaler.pkl`: Objek penyesuaian skala data numerik.
   * `feature_metadata.pkl`: Berkas informasi tentang daftar kolom, tipe data, serta nilai minimum/maksimum dari data training untuk menghasilkan elemen antarmuka web secara otomatis.
2. **Deserialisasi**:
   Aplikasi Streamlit (`app.py`) memuat kembali berkas-berkas `.pkl` tersebut untuk memproses input baru dari pengguna dan menghasilkan prediksi secara instan.

### B. Cara Menjalankan Modul 3
1. Pastikan Anda telah melatih model terlebih dahulu dengan menjalankan `quick_start.py`. Hal ini diperlukan agar berkas `.pkl` terbentuk di dalam folder `results/models/`.
2. (Opsional) Jalankan uji integrasi deployment untuk memastikan proses memuat model berfungsi dengan benar:
   ```bash
   python test_deployment.py
   ```
3. Jalankan aplikasi web Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Buka peramban (browser) Anda dan akses alamat lokal yang diberikan (biasanya `http://localhost:8501`).

---

## 📖 Panduan Tambahan
Untuk penjelasan yang lebih detail mengenai isi kelas dan metode dari setiap modul di dalam folder `src/`, silakan merujuk ke berkas [SRC_PENJELASAN.md](SRC_PENJELASAN.md).
