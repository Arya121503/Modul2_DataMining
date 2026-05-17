"""
data_loader.py
Modul untuk membaca dan memuat dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Kelas untuk memuat dan membaca data"""
    
    def __init__(self, data_path):
        """
        Inisialisasi DataLoader
        
        Parameters:
        -----------
        data_path : str
            Path ke folder data
        """
        self.data_path = Path(data_path)
    
    def load_csv(self, filename):
        """
        Membaca file CSV
        
        Parameters:
        -----------
        filename : str
            Nama file CSV yang akan dibaca
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dari file CSV
        """
        try:
            file_path = self.data_path / filename
            df = pd.read_csv(file_path)
            print(f"✓ Data berhasil dimuat: {filename}")
            print(f"  Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"✗ File tidak ditemukan: {filename}")
            return None
    
    def get_data_info(self, df):
        """
        Menampilkan informasi umum tentang dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame untuk dianalisis
        """
        print("\n" + "="*50)
        print("DATA INFORMATION")
        print("="*50)
        print(f"Shape: {df.shape}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\nDeskripsi Statistik:\n{df.describe()}")
        print("="*50 + "\n")
    
    def check_duplicates(self, df):
        """
        Mengecek duplikat di dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame untuk dicek
        """
        duplicates = df.duplicated().sum()
        print(f"Jumlah duplikat: {duplicates}")
        return duplicates

if __name__ == "__main__":
    # Contoh penggunaan
    loader = DataLoader("data/raw")
    df = loader.load_csv("dataset.csv")
    
    if df is not None:
        loader.get_data_info(df)
        loader.check_duplicates(df)
