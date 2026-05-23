import os
import sys
import pickle
import pandas as pd

# Put workspace root into path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import DataPreprocessor

def test_inference():
    print("🧪 Starting Deployment Deserialization Tests...")
    
    # Paths to serialized objects
    model_path = 'results/models/best_model.pkl'
    prep_path = 'results/models/preprocessor.pkl'
    scaler_path = 'results/models/scaler.pkl'
    meta_path = 'results/models/feature_metadata.pkl'
    
    # Assert they all exist
    for p in [model_path, prep_path, scaler_path, meta_path]:
        assert os.path.exists(p), f"Missing file: {p}"
        print(f"  ✓ Found serialized artifact: {p}")
        
    # Load them
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(prep_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
        
    print("  ✓ Deserialized all components successfully.")
    
    features = metadata['features']
    target = metadata['target']
    details = metadata['details']
    
    print(f"  ✓ Target label: {target}")
    print(f"  ✓ Features trained: {features}")
    
    # Construct a sample observation based on features list
    # Let's verify details
    sample_obs = {}
    for feat in features:
        feat_info = details[feat]
        if feat_info['type'] == 'categorical':
            # Pick first category
            sample_obs[feat] = feat_info['categories'][0]
        else:
            # Pick mean value
            sample_obs[feat] = feat_info['mean']
            
    print(f"  ✓ Sample test input constructed: {sample_obs}")
    
    # Convert to DataFrame
    df_input = pd.DataFrame([sample_obs])
    
    # Preprocess categorical using preprocessor
    categorical_cols = [f for f, info in details.items() if info['type'] == 'categorical']
    for col in categorical_cols:
        le = preprocessor.label_encoders[col]
        df_input[col] = le.transform(df_input[col].astype(str))
        
    print(f"  ✓ Categorical columns encoded: {df_input.iloc[0].to_dict()}")
    
    # Scale using scaler
    df_scaled = scaler.normalize_scale(df_input, df_input.columns, method='standard', fit=False)
    print(f"  ✓ Scaled features: {df_scaled.iloc[0].to_dict()}")
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    print(f"  🎉 Prediction Successful! Outcome: {prediction}")
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(df_scaled)[0]
        print(f"  📊 Prediction Probabilities: {probabilities}")
        
    print("🚀 Deployment verification completed successfully!")

if __name__ == "__main__":
    test_inference()
