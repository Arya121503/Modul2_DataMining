# START HERE - Panduan Awal Praktikum

Project ini dibuat supaya mahasiswa bisa langsung clone, install, lalu menjalankan praktikum tanpa langkah tambahan yang rumit.

## Langkah Cepat

1. Clone repository ini.
2. Buka terminal di folder project.
3. Buat dan aktifkan virtual environment.
4. Install dependencies dari requirements.
5. Jalankan demo dengan `python quick_start.py`.
6. Buka notebook kalau ingin eksplorasi manual.

## Jika Baru Pertama Kali

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python quick_start.py
```

## Alur Belajar yang Disarankan

1. Baca [README.md](README.md) untuk ringkasan project dan dataset yang cocok.
2. Jalankan `quick_start.py` untuk memastikan environment sudah benar.
3. Buka notebook secara berurutan:
   1. [notebooks/1_eda.ipynb](notebooks/1_eda.ipynb)
   2. [notebooks/2_preprocessing.ipynb](notebooks/2_preprocessing.ipynb)
   3. [notebooks/3_modeling.ipynb](notebooks/3_modeling.ipynb)

## Catatan Dataset

- Demo utama sekarang memakai `data/raw/customer_purchase_data.csv`.
- Kalau ingin memakai dataset lain, simpan file CSV di `data/raw/dataset.csv` atau sesuaikan nama file di notebook.
- Dataset terbaik adalah yang punya kolom numerik dan kategorikal.
- Untuk klasifikasi, target biner biasanya paling aman untuk latihan awal.

## Modul Utama

- `src/data_loader.py` untuk membaca data.
- `src/preprocessing.py` untuk cleaning dan encoding.
- `src/model.py` untuk training dan tuning.
- `src/evaluation.py` untuk evaluasi model.

## File yang Perlu Diperhatikan

- [README.md](README.md) untuk dokumentasi utama.
- [ALUR_PRAKTIKUM.md](ALUR_PRAKTIKUM.md) untuk ringkasan alur praktikum.
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) untuk detail struktur file.

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
