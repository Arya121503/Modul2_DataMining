# 2 Preprocessing

# 2 — Data Preprocessing
Notebook ini memuat data raw, melakukan preprocessing dasar menggunakan modul `src/`, lalu menyimpan hasil ke `data/processed/dataset_clean.csv`.

```python
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / 'src').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor

RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
RAW_FILE = RAW_DIR / 'dataset.csv'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_FILE
```

```python
# Load raw data
loader = DataLoader(str(RAW_DIR))
df = loader.load_csv('dataset.csv')
df.head()
```

```python
# Basic preprocessing
pre = DataPreprocessor()

# 1) Handle missing values
df_clean = pre.handle_missing_values(df, strategy='mean')

# 2) Encode categorical columns
cat_cols = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
df_clean = pre.encode_categorical(df_clean, cat_cols, encode_type='label')

df_clean.head()
```

```python
# (Opsional) Hapus outliers numerik (IQR)
# Catatan: Ini bisa mengurangi jumlah baris data.
num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
# Jika ada target, jangan dipakai untuk outlier removal
preferred_targets = ['purchased', 'loan_approved', 'target', 'label']
target_col = next((c for c in preferred_targets if c in df_clean.columns), None)
if target_col in num_cols:
    num_cols = [c for c in num_cols if c != target_col]

# Uncomment jika ingin menghapus outliers
# df_clean = pre.handle_outliers(df_clean, num_cols, method='iqr', threshold=1.5)

df_clean.shape
```

```python
# Save processed dataset
out_path = PROCESSED_DIR / 'dataset_clean.csv'
df_clean.to_csv(out_path, index=False)
print('Saved:', out_path)
out_path
```

## Next
Lanjut ke notebook `3_modeling.ipynb` untuk training & evaluasi model.
