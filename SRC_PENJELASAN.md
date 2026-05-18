# Penjelasan Folder `src`

Dokumen ini menjelaskan isi folder `src` dengan bahasa sederhana agar mahasiswa baru bisa memahami alur program tanpa harus membaca kode mentah sekaligus.

## Gambaran Umum

Folder `src` berisi kode inti project. Fungsi-fungsinya dipisah agar:

- lebih mudah dibaca,
- lebih mudah dipakai ulang di notebook,
- dan lebih gampang dirawat kalau ada perubahan.

Urutan logisnya adalah:

1. baca data,
2. bersihkan data,
3. olah fitur,
4. latih model,
5. evaluasi hasil.

## 1. `src/__init__.py`

Fungsi file ini:

- menandai folder `src` sebagai Python package,
- memudahkan import seperti `from src import DataLoader`.

Isi pentingnya:

- `from .data_loader import DataLoader` berarti kelas `DataLoader` diambil dari file `data_loader.py`.
- `__all__` berisi daftar objek yang boleh di-import langsung dari `src`.

Intinya:

- file ini tidak menjalankan proses data,
- hanya membuat folder `src` lebih rapi untuk dipakai sebagai package.

## 2. `src/data_loader.py`

File ini tugasnya membaca data dari file CSV.

### Bagian import

- `import pandas as pd` untuk membaca dan mengolah tabel data.
- `import numpy as np` disiapkan untuk operasi numerik, walau di file ini belum banyak dipakai.
- `from pathlib import Path` untuk menangani path file dengan aman.

### Class `DataLoader`

Class ini dibuat supaya semua fungsi terkait loading data dikelompokkan dalam satu tempat.

### `__init__(self, data_path)`

- menyimpan lokasi folder data ke `self.data_path`,
- jadi nanti file bisa dibaca dari folder yang sama tanpa menulis path panjang berulang-ulang.

### `load_csv(self, filename)`

Bagian ini:

- menggabungkan folder data dan nama file,
- membaca CSV dengan `pd.read_csv()`,
- menampilkan pesan sukses dan shape data,
- jika file tidak ada, menampilkan pesan error dan mengembalikan `None`.

### `get_data_info(self, df)`

Fungsi ini menampilkan info awal dataset:

- shape,
- tipe data tiap kolom,
- jumlah missing value,
- statistik deskriptif.

Ini berguna untuk EDA awal sebelum modeling.

### `check_duplicates(self, df)`

Fungsi ini menghitung jumlah baris duplikat.

Kenapa penting:

- data duplikat bisa memengaruhi analisis,
- terutama kalau ada baris yang muncul lebih dari sekali tanpa alasan.

## 3. `src/preprocessing.py`

File ini menangani pembersihan dan transformasi data.

### Import utama

- `StandardScaler` dan `MinMaxScaler` dipakai untuk scaling,
- `LabelEncoder` dipakai untuk mengubah kategori menjadi angka.

### Class `DataPreprocessor`

Class ini menyimpan objek preprocessing agar bisa dipakai ulang.

### `__init__(self)`

- membuat scaler standar,
- membuat scaler min-max,
- menyiapkan dictionary untuk label encoder.

### `handle_missing_values(self, df, strategy='mean', columns=None)`

Ini fungsi untuk menangani nilai kosong.

Alurnya:

- salin data dulu supaya data asli tidak berubah,
- tentukan kolom yang punya missing value,
- isi data kosong berdasarkan strategi yang dipilih.

Strategi yang tersedia:

- `mean` untuk angka,
- `median` untuk angka,
- `drop` untuk menghapus baris yang kosong,
- `forward_fill` untuk mengisi dengan nilai sebelumnya.

Kalau kolomnya bukan numerik, fungsi ini memakai nilai yang paling sering muncul (mode).

### `handle_outliers(self, df, columns, method='iqr', threshold=1.5)`

Fungsi ini menghapus outlier menggunakan metode IQR.

Langkahnya:

- hitung Q1 dan Q3,
- hitung IQR,
- tentukan batas bawah dan atas,
- hapus baris yang keluar dari batas tersebut.

### `encode_categorical(self, df, columns, encode_type='label')`

Fungsi ini mengubah data kategori menjadi angka.

Pilihan encoding:

- `label` untuk mengubah kategori menjadi 0, 1, 2, dan seterusnya,
- `onehot` untuk membuat kolom baru per kategori.

Untuk mahasiswa baru, label encoding lebih mudah dipahami, tetapi one-hot lebih aman untuk banyak kasus model.

### `normalize_scale(self, df, columns, method='standard', fit=True)`

Fungsi ini menyesuaikan skala angka.

Kenapa penting:

- fitur dengan skala besar bisa mendominasi model tertentu,
- scaling membantu model melihat data secara lebih seimbang.

Parameter `fit`:

- `True` berarti scaler dilatih dulu pada data train,
- `False` berarti data baru hanya di-transform dengan scaler yang sudah dilatih.

Ini penting supaya tidak terjadi data leakage.

## 4. `src/feature_engineering.py`

File ini dipakai untuk membuat fitur baru atau memilih fitur terbaik.

### Import utama

- `SelectKBest` untuk memilih fitur,
- `f_classif` dan `mutual_info_classif` untuk menilai fitur,
- `PCA` untuk mereduksi dimensi.

### Class `FeatureEngineer`

Semua fungsi feature engineering dikelompokkan di sini.

### `create_interaction_features(self, df, col1, col2)`

- membuat kolom baru dari hasil perkalian dua fitur,
- berguna kalau hubungan antar fitur penting.

### `create_polynomial_features(self, df, col, degree=2)`

- membuat fitur pangkat dari kolom tertentu,
- contoh: `age_pow2` dari `age ** 2`.

### `create_binning_features(self, df, col, bins=5, labels=None)`

- mengelompokkan angka ke dalam rentang tertentu,
- misalnya umur dibagi menjadi kategori muda, dewasa, dan senior.

### `select_best_features(self, X, y, k=10, method='selectkbest')`

- memilih fitur paling relevan terhadap target,
- hasilnya berupa daftar nama fitur terbaik.

### `apply_pca(self, X, n_components=0.95)`

- mengubah fitur banyak menjadi lebih sedikit,
- tetapi tetap mempertahankan sebagian besar variasi data.

### `get_feature_importance(self, model, feature_names)`

- mengambil tingkat pentingnya fitur dari model,
- bekerja jika model punya atribut `feature_importances_` atau `coef_`.

## 5. `src/model.py`

File ini mengurus pembagian data, training model, validasi, tuning, dan penyimpanan model.

### Import utama

- `train_test_split` untuk membagi data,
- `cross_val_score` untuk validasi silang,
- `GridSearchCV` untuk tuning parameter,
- beberapa model klasifikasi dari scikit-learn.

### Class `ModelBuilder`

Class ini menyimpan semua proses modeling.

### `__init__(self)`

- menyiapkan dictionary model,
- menyiapkan variabel untuk best model.

### `split_data(self, X, y, test_size=0.2, random_state=42, stratify=None)`

- membagi data menjadi train dan test,
- menampilkan jumlah data di tiap bagian.

Kenapa penting:

- model dilatih di train,
- hasilnya diuji di test,
- supaya penilaian lebih jujur.

### `train_logistic_regression(self, X_train, y_train)`

- melatih model Logistic Regression,
- cocok untuk klasifikasi dasar dan mudah dijelaskan.

### `train_random_forest(self, X_train, y_train, n_estimators=100)`

- melatih Random Forest,
- biasanya lebih kuat dari model linear untuk data yang lebih kompleks.

### `train_gradient_boosting(self, X_train, y_train, n_estimators=100)`

- melatih Gradient Boosting,
- cocok untuk membandingkan performa model lain.

### `train_svm(self, X_train, y_train, kernel='rbf')`

- melatih Support Vector Machine,
- dipakai kalau ingin mencoba model lain selain pohon keputusan.

### `cross_validate(self, model, X, y, cv=5)`

- mengecek performa model dengan beberapa pembagian data,
- hasilnya lebih stabil daripada satu kali split saja.

### `hyperparameter_tuning(self, X_train, y_train, model_name='random_forest', params=None)`

- mencari kombinasi parameter terbaik dengan GridSearchCV,
- mengembalikan model terbaik.

### `save_model(self, model, filename)`

- menyimpan model ke file `.pkl`.

### `load_model(self, filename)`

- memuat model dari file yang sudah disimpan.

## 6. `src/evaluation.py`

File ini dipakai untuk menilai hasil model.

### Import utama

- metric classification dan regression dari `sklearn.metrics`,
- `matplotlib` dan `seaborn` untuk visualisasi.

### Class `ModelEvaluator`

Kelas ini menyimpan fungsi evaluasi model.

### `evaluate_classification(self, y_true, y_pred, y_pred_proba=None)`

- menghitung accuracy,
- precision,
- recall,
- F1-score,
- dan ROC-AUC jika probabilitas tersedia.

### `evaluate_regression(self, y_true, y_pred)`

- menghitung MSE,
- RMSE,
- MAE,
- dan R².

### `get_classification_report(self, y_true, y_pred, target_names=None)`

- menampilkan laporan klasifikasi detail.

### `get_confusion_matrix(self, y_true, y_pred)`

- mengembalikan confusion matrix dalam bentuk array.

### `plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6))`

- menampilkan confusion matrix dalam bentuk heatmap.

### `plot_roc_curve(self, y_true, y_pred_proba, figsize=(8, 6))`

- menampilkan kurva ROC,
- berguna untuk melihat trade-off true positive dan false positive.

### `plot_feature_importance(self, feature_importance_df, top_n=10, figsize=(10, 6))`

- menampilkan fitur paling penting.

### `plot_predictions_vs_actual(self, y_true, y_pred, figsize=(8, 6))`

- dipakai untuk regresi,
- membandingkan hasil prediksi dengan nilai asli.

### `compare_models(self, models_dict, X_test, y_test)`

- membandingkan beberapa model berdasarkan akurasi.

## Ringkasan Sederhana untuk Mahasiswa

Kalau disederhanakan, folder `src` bekerja seperti ini:

- `data_loader.py` = ambil data,
- `preprocessing.py` = bersihkan data,
- `feature_engineering.py` = ubah atau pilih fitur,
- `model.py` = latih model,
- `evaluation.py` = nilai model.

Kalau mahasiswa baru, fokuskan dulu ke urutan itu sebelum masuk ke tuning atau feature engineering yang lebih rumit.
