"""
feature_engineering.py
Modul untuk feature engineering dan feature selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

class FeatureEngineer:
    """Kelas untuk feature engineering"""
    
    def __init__(self):
        """Inisialisasi FeatureEngineer"""
        self.pca = None
    
    def create_interaction_features(self, df, col1, col2):
        """
        Membuat interaction features
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang akan diproses
        col1, col2 : str
            Nama kolom untuk interaction
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan interaction feature
        """
        df_copy = df.copy()
        new_col = f"{col1}_x_{col2}"
        df_copy[new_col] = df_copy[col1] * df_copy[col2]
        print(f"✓ Interaction feature dibuat: {new_col}")
        return df_copy
    
    def create_polynomial_features(self, df, col, degree=2):
        """
        Membuat polynomial features
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang akan diproses
        col : str
            Nama kolom
        degree : int
            Derajat polynomial
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan polynomial features
        """
        df_copy = df.copy()
        for d in range(2, degree + 1):
            new_col = f"{col}_pow{d}"
            df_copy[new_col] = df_copy[col] ** d
            print(f"✓ Polynomial feature dibuat: {new_col}")
        return df_copy
    
    def create_binning_features(self, df, col, bins=5, labels=None):
        """
        Membuat binning features (discretization)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame yang akan diproses
        col : str
            Nama kolom
        bins : int
            Jumlah bins
        labels : list
            Label untuk bins
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan binning feature
        """
        df_copy = df.copy()
        new_col = f"{col}_binned"
        df_copy[new_col] = pd.cut(df_copy[col], bins=bins, labels=labels)
        print(f"✓ Binning feature dibuat: {new_col}")
        return df_copy
    
    def select_best_features(self, X, y, k=10, method='selectkbest'):
        """
        Memilih feature terbaik
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        k : int
            Jumlah features yang dipilih
        method : str
            Metode: 'selectkbest' atau 'mutual_info'
            
        Returns:
        --------
        list
            Nama features yang dipilih
        """
        if method == 'selectkbest':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"✓ {k} best features dipilih menggunakan {method}")
        print(f"  Features: {selected_features}")
        
        return selected_features
    
    def apply_pca(self, X, n_components=0.95):
        """
        Menerapkan PCA untuk dimensionality reduction
        
        Parameters:
        -----------
        X : pd.DataFrame atau array
            Features
        n_components : float atau int
            Jumlah atau proporsi components
            
        Returns:
        --------
        array
            Features setelah PCA
        """
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"✓ PCA diterapkan")
        print(f"  N Components: {self.pca.n_components_}")
        print(f"  Explained Variance: {explained_var:.2%}")
        
        return X_pca
    
    def get_feature_importance(self, model, feature_names):
        """
        Mendapatkan feature importance dari model
        
        Parameters:
        -----------
        model : sklearn model
            Model yang sudah dilatih
        feature_names : list
            Nama-nama features
            
        Returns:
        --------
        pd.DataFrame
            Feature importance sorted
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("✗ Model tidak memiliki feature importance")
            return None
        
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("✓ Feature importance:")
        print(feature_imp.head(10))
        
        return feature_imp

if __name__ == "__main__":
    # Contoh penggunaan
    from data_loader import DataLoader
    
    loader = DataLoader("data/raw")
    df = loader.load_csv("dataset.csv")
    
    engineer = FeatureEngineer()
    
    # Buat interaction features (contoh)
    # df = engineer.create_interaction_features(df, 'feature1', 'feature2')
    
    # Buat polynomial features
    # df = engineer.create_polynomial_features(df, 'feature1', degree=2)
    
    # Buat binning features
    # df = engineer.create_binning_features(df, 'age', bins=5)
    
    print("✓ Feature engineering selesai!")
