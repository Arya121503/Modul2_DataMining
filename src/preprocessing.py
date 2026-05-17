"""
preprocessing.py
Modul untuk data cleaning dan preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class DataPreprocessor:
    """Kelas untuk preprocessing data"""
    
    def __init__(self):
        """Inisialisasi preprocessor"""
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
    
    def handle_missing_values(self, df, strategy='mean', columns=None):
        """
        Menangani missing values
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang akan diproses
        strategy : str
            Strategi: 'mean', 'median', 'drop', 'forward_fill'
        columns : list
            Kolom yang akan diproses (default: semua)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame setelah handling missing values
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.columns[df_copy.isnull().any()]
        
        for col in columns:
            if df_copy[col].isnull().sum() > 0:
                if strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    else:
                        mode_val = df_copy[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if not mode_val.empty else ""
                        df_copy[col].fillna(fill_val, inplace=True)
                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
                    else:
                        mode_val = df_copy[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if not mode_val.empty else ""
                        df_copy[col].fillna(fill_val, inplace=True)
                elif strategy == 'drop':
                    df_copy.dropna(subset=[col], inplace=True)
                elif strategy == 'forward_fill':
                    df_copy[col].fillna(method='ffill', inplace=True)
                
                print(f"✓ Missing values di '{col}' ditangani dengan {strategy}")
        
        return df_copy
    
    def handle_outliers(self, df, columns, method='iqr', threshold=1.5):
        """
        Menangani outliers
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang akan diproses
        columns : list
            Kolom untuk deteksi outliers
        method : str
            Metode: 'iqr', 'zscore'
        threshold : float
            Threshold untuk outliers
            
        Returns:
        --------
        pd.DataFrame
            DataFrame setelah handling outliers
        """
        df_copy = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                outliers = ((df_copy[col] < lower) | (df_copy[col] > upper)).sum()
                df_copy = df_copy[(df_copy[col] >= lower) & (df_copy[col] <= upper)]
                print(f"✓ {outliers} outliers di '{col}' dihapus (IQR method)")
        
        return df_copy
    
    def encode_categorical(self, df, columns, encode_type='label'):
        """
        Encode variabel kategori
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang akan diproses
        columns : list
            Kolom kategori untuk di-encode
        encode_type : str
            Tipe encoding: 'label' atau 'onehot'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame setelah encoding
        """
        df_copy = df.copy()
        
        for col in columns:
            if encode_type == 'label':
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                self.label_encoders[col] = le
                print(f"✓ '{col}' di-encode menggunakan Label Encoding")
            
            elif encode_type == 'onehot':
                df_copy = pd.get_dummies(df_copy, columns=[col], prefix=col)
                print(f"✓ '{col}' di-encode menggunakan One-Hot Encoding")
        
        return df_copy
    
    def normalize_scale(self, df, columns, method='standard', fit=True):
        """
        Normalisasi/Scale data
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang akan diproses
        columns : list
            Kolom untuk di-scale
        method : str
            Metode: 'standard' atau 'minmax'
        fit : bool
            Jika True: fit_transform, jika False: transform saja (pakai scaler yang sudah di-fit)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame setelah scaling
        """
        df_copy = df.copy()
        
        if method == 'standard':
            if fit:
                df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
            else:
                df_copy[columns] = self.scaler.transform(df_copy[columns])
            print(f"✓ Data di-scale menggunakan Standard Scaler")
        
        elif method == 'minmax':
            if fit:
                df_copy[columns] = self.minmax_scaler.fit_transform(df_copy[columns])
            else:
                df_copy[columns] = self.minmax_scaler.transform(df_copy[columns])
            print(f"✓ Data di-scale menggunakan MinMax Scaler")
        
        return df_copy

if __name__ == "__main__":
    # Contoh penggunaan
    from data_loader import DataLoader
    
    loader = DataLoader("data/raw")
    df = loader.load_csv("dataset.csv")
    
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, strategy='mean')
    
    # Handle outliers
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = preprocessor.handle_outliers(df, numeric_cols)
    
    # Encode categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = preprocessor.encode_categorical(df, categorical_cols, encode_type='label')
    
    # Scale data
    df = preprocessor.normalize_scale(df, numeric_cols, method='standard')
    
    print("\n✓ Preprocessing selesai!")
    print(df.head())
