# 1 Eda

# 1 — Exploratory Data Analysis (EDA)
Notebook ini membaca `data/raw/dataset.csv` lalu melakukan EDA dasar (ringkasan, missing values, distribusi, korelasi).

```python
# Setup path + imports
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_theme()

ROOT = Path.cwd().resolve()
while not (ROOT / 'src').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / 'data' / 'raw' / 'customer_purchase_data.csv'
DATASET_PATH
```

```python
# Load dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(f'Dataset tidak ditemukan: {DATASET_PATH}')

df = pd.read_csv(DATASET_PATH)
print('Shape:', df.shape)
df.head()
```

```python
# Info umum
display(df.info())
df.describe(include='all')
```

```python
# Missing values & duplicates
missing = df.isna().sum().sort_values(ascending=False)
display(missing[missing > 0])
print('Duplicate rows:', df.duplicated().sum())
```

```python
# Distribusi kolom numerik
num_cols = df.select_dtypes(include=['number']).columns.tolist()
df[num_cols].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()
```

```python
# Korelasi (numerik)
if len(num_cols) >= 2:
    corr = df[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Correlation Heatmap (Numeric Columns)')
    plt.tight_layout()
    plt.show()
else:
    print('Tidak cukup kolom numerik untuk korelasi.')
```
