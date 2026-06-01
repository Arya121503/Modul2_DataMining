# 3 Modeling

# 3 — Modeling & Evaluation
Notebook ini memuat `data/processed/dataset_clean.csv`, split train/test, scaling, training model, evaluasi, dan menyimpan model terbaik.

```python
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().resolve()
while not (ROOT / 'src').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import DataPreprocessor
from src.model import ModelBuilder
from src.evaluation import ModelEvaluator

PROCESSED_PATH = ROOT / 'data' / 'processed' / 'dataset_clean.csv'
MODEL_OUT = ROOT / 'results' / 'models' / 'best_model.pkl'
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH
```

```python
# Load processed data
if not PROCESSED_PATH.exists():
    raise FileNotFoundError('dataset_clean.csv belum ada. Jalankan notebook 2_preprocessing dulu.')

df = pd.read_csv(PROCESSED_PATH)
print('Shape:', df.shape)
df.head()
```

```python
# Prepare features & target
preferred_targets = ['purchased', 'loan_approved', 'target', 'label']
target_col = next((c for c in preferred_targets if c in df.columns), None)
if target_col is None:
    target_col = df.columns[-1]
    print('⚠️ Target column tidak ditemukan di daftar default; memakai kolom terakhir:', target_col)
else:
    print('✓ Target column:', target_col)

X = df.drop(columns=[target_col])
y = df[target_col]

X.shape, y.shape
```

```python
# Split + scaling (fit on train, transform on test)
builder = ModelBuilder()
X_train, X_test, y_train, y_test = builder.split_data(X, y, test_size=0.2)

scaler = DataPreprocessor()
X_train_s = scaler.normalize_scale(X_train, X_train.columns, method='standard', fit=True)
X_test_s = scaler.normalize_scale(X_test, X_test.columns, method='standard', fit=False)

X_train_s.head()
```

```python
# Train models
model_lr = builder.train_logistic_regression(X_train_s, y_train)
model_rf = builder.train_random_forest(X_train_s, y_train, n_estimators=200)
```

```python
# Evaluate
evaluator = ModelEvaluator()

y_pred_lr = model_lr.predict(X_test_s)
y_pred_rf = model_rf.predict(X_test_s)

print('Logistic Regression:')
m_lr = evaluator.evaluate_classification(y_test, y_pred_lr)

print('Random Forest:')
m_rf = evaluator.evaluate_classification(y_test, y_pred_rf)

comparison = evaluator.compare_models({'LogReg': model_lr, 'RandomForest': model_rf}, X_test_s, y_test)
comparison
```

```python
# Save best model
best_model = model_rf if m_rf.get('accuracy', 0) >= m_lr.get('accuracy', 0) else model_lr
builder.save_model(best_model, str(MODEL_OUT))
MODEL_OUT
```

## Selesai
- Model terbaik tersimpan di `results/models/best_model.pkl`
- Anda bisa lanjut melakukan tuning dengan `ModelBuilder.hyperparameter_tuning()` jika diperlukan.
