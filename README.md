# Modul 2 - Pemodelan Data

Project ini berisi alur praktikum data mining yang sudah disiapkan untuk langsung dicoba mahasiswa.

## Quick Start

```bash
cd modul2_datamining
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python quick_start.py
```

Kalau ingin eksplorasi interaktif, buka [notebooks/1_eda.ipynb](notebooks/1_eda.ipynb).

## Isi Project

- `data/raw/` untuk data mentah. File default adalah `dataset.csv`.
- `notebooks/` untuk praktik EDA, preprocessing, dan modeling.
- `src/` untuk modul utama yang dipakai ulang oleh notebook dan script.
- `results/` untuk output model dan hasil analisis saat dijalankan lokal.

## Dataset yang Disarankan

Project ini paling aman dipakai dengan dataset yang:

- punya kolom numerik dan kategorikal,
- punya target klasifikasi biner,
- ukurannya tidak terlalu besar.

Contoh dataset yang cocok:

| Dataset | Cocok untuk | Catatan |
|---------|-------------|---------|
| Titanic | klasifikasi biner | mudah dipahami, banyak contoh online |
| Breast Cancer Wisconsin | klasifikasi biner | cepat, cocok untuk demo |
| Adult Income | klasifikasi biner | perlu encoding kategorikal |
| Bank Marketing | klasifikasi biner | bagus untuk latihan evaluasi |

Kalau memakai dataset sendiri, simpan sebagai `data/raw/dataset.csv` atau ubah nama file di notebook.

## Notebook yang Tersedia

1. [notebooks/1_eda.ipynb](notebooks/1_eda.ipynb) - eksplorasi data.
2. [notebooks/2_preprocessing.ipynb](notebooks/2_preprocessing.ipynb) - cleaning dan encoding.
3. [notebooks/3_modeling.ipynb](notebooks/3_modeling.ipynb) - split, training, dan evaluasi.

## Modul Utama di src

- `src/data_loader.py` untuk membaca data.
- `src/preprocessing.py` untuk handling missing value, encoding, dan scaling.
- `src/model.py` untuk split data, training, cross-validation, dan GridSearchCV.
- `src/evaluation.py` untuk accuracy, precision, recall, F1, confusion matrix, dan ROC curve.

## Library Inti

- pandas, numpy, scipy
- matplotlib, seaborn, plotly
- scikit-learn
- jupyter, ipython

## Catatan Penting

- Data asli tetap di `data/raw/`.
- Hasil proses akan dibuat saat notebook dijalankan.
- Kalau ingin versi penjelasan yang lebih singkat, buka [START_HERE.md](START_HERE.md).

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
