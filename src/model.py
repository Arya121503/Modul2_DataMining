"""
model.py
Modul untuk building dan training model machine learning
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelBuilder:
    """Kelas untuk building dan training model"""
    
    def __init__(self):
        """Inisialisasi ModelBuilder"""
        self.models = {}
        self.best_model = None
        self.scaler = None
    
    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=None):
        """
        Split data menjadi train dan test set
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        test_size : float
            Proporsi test set
        random_state : int
            Random seed
        stratify : array-like, optional
            Untuk stratified split
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        if stratify is not None:
            stratify = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        print(f"✓ Data split selesai")
        print(f"  Train set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Melatih Logistic Regression
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
            
        Returns:
        --------
        model
            Trained model
        """
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        print("✓ Logistic Regression model dilatih")
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """
        Melatih Random Forest
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        n_estimators : int
            Jumlah trees
            
        Returns:
        --------
        model
            Trained model
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        print(f"✓ Random Forest model dilatih ({n_estimators} trees)")
        return model
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators=100):
        """
        Melatih Gradient Boosting
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        n_estimators : int
            Jumlah iterations
            
        Returns:
        --------
        model
            Trained model
        """
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        
        print(f"✓ Gradient Boosting model dilatih ({n_estimators} iterations)")
        return model
    
    def train_svm(self, X_train, y_train, kernel='rbf'):
        """
        Melatih Support Vector Machine
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        kernel : str
            Kernel type: 'linear', 'rbf', 'poly'
            
        Returns:
        --------
        model
            Trained model
        """
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train, y_train)
        self.models['svm'] = model
        
        print(f"✓ SVM model dilatih (kernel={kernel})")
        return model
    
    def cross_validate(self, model, X, y, cv=5):
        """
        Melakukan cross-validation
        
        Parameters:
        -----------
        model : sklearn model
            Model untuk di-validate
        X : array-like
            Features
        y : array-like
            Target
        cv : int
            Jumlah folds
            
        Returns:
        --------
        scores : array
            Cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"✓ Cross-validation ({cv}-fold) selesai")
        print(f"  Mean Score: {scores.mean():.4f}")
        print(f"  Std Dev: {scores.std():.4f}")
        
        return scores
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest', params=None):
        """
        Hyperparameter tuning menggunakan GridSearchCV
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        model_name : str
            Tipe model
        params : dict
            Parameter grid untuk tuning
            
        Returns:
        --------
        model
            Best model setelah tuning
        """
        if model_name == 'random_forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            if params is None:
                params = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
        
        elif model_name == 'logistic_regression':
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            if params is None:
                params = {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
        
        else:
            print(f"✗ Model {model_name} tidak tersedia")
            return None
        
        grid_search = GridSearchCV(
            base_model, params, cv=5, n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Hyperparameter tuning selesai untuk {model_name}")
        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"  Best Score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def save_model(self, model, filename):
        """
        Simpan model ke file
        
        Parameters:
        -----------
        model : sklearn model
            Model untuk disimpan
        filename : str
            Path file untuk menyimpan model
        """
        import os
        # Membuat folder parent jika belum ada untuk menghindari FileNotFoundError
        parent_dir = os.path.dirname(filename)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model disimpan ke: {filename}")
    
    def load_model(self, filename):
        """
        Load model dari file
        
        Parameters:
        -----------
        filename : str
            Path file model
            
        Returns:
        --------
        model
            Loaded model
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model di-load dari: {filename}")
        return model

if __name__ == "__main__":
    # Contoh penggunaan akan dilakukan di notebook
    print("Model builder module ready!")
