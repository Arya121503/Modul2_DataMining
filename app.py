import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Ensure project root is in path so pickle can load classes from src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import DataPreprocessor

# Set page config
st.set_page_config(
    page_title="Model Deployment Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injected Premium CSS for Glassmorphic Dark Mode Theme
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Apply fonts globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    h1, h2, h3, .stHeader {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Premium Header Container */
    .header-container {
        padding: 2.5rem;
        background: radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 100% 100%, rgba(168, 85, 247, 0.08) 0%, transparent 50%);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
    }
    
    /* Card design */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }
    
    /* Custom predictions card */
    .result-card-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.03) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.15);
        text-align: center;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        animation: float 4s ease-in-out infinite;
    }
    
    .result-card-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.03) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.15);
        text-align: center;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        animation: float 4s ease-in-out infinite;
    }
    
    /* Float animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    /* Stylize st.button */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #7c3aed 100%);
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        cursor: pointer;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
    }

    /* Input focus colors */
    input, select, textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Info box overrides */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load serialized objects
@st.cache_resource
def load_deployment_artifacts():
    paths = {
        'model': 'results/models/best_model.pkl',
        'preprocessor': 'results/models/preprocessor.pkl',
        'scaler': 'results/models/scaler.pkl',
        'metadata': 'results/models/feature_metadata.pkl'
    }
    
    artifacts = {}
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                artifacts[key] = pickle.load(f)
        else:
            st.error(f"Missing deployment file: `{path}`. Please run `quick_start.py` to train and serialize the models.")
            return None
    return artifacts

# Load the artifacts
artifacts = load_deployment_artifacts()

if artifacts:
    model = artifacts['model']
    preprocessor = artifacts['preprocessor']
    scaler = artifacts['scaler']
    metadata = artifacts['metadata']
    
    features_list = metadata['features']
    target_name = metadata['target']
    details = metadata['details']
    
    # Premium Layout Header
    st.markdown(f"""
    <div class="header-container">
        <h1 style='margin: 0; font-size: 2.8rem;'>🧠 AI Customer Prediction Engine</h1>
        <p style='margin: 5px 0 0 0; color: #94a3b8; font-size: 1.1rem;'>
            Predict target class <b>'{target_name}'</b> using the serialized <b>{type(model).__name__}</b> model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two primary columns: left for inputs, right for results
    col_input, col_result = st.columns([3, 2], gap="large")
    
    with col_input:
        st.markdown("<h3 style='margin-bottom: 1rem; color: #38bdf8;'>⚙️ Input Customer Features</h3>", unsafe_allow_html=True)
        
        # We wrap input controls in a glass container
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        input_data = {}
        
        # Dynamically generate fields
        for feat in features_list:
            feat_info = details[feat]
            
            # Format feature name for display (capitalize and replace underscores)
            label = feat.replace('_', ' ').title()
            
            if feat_info['type'] == 'categorical':
                categories = feat_info['categories']
                input_data[feat] = st.selectbox(
                    f"Select {label}",
                    options=categories,
                    key=f"input_{feat}"
                )
            else:
                # Numeric feature
                val_min = feat_info['min']
                val_max = feat_info['max']
                val_mean = feat_info['mean']
                
                # Check bounds types and calculate proper steps
                is_int = all(isinstance(v, (int, np.integer)) for v in [val_min, val_max, val_mean])
                step = 1 if is_int else 0.1
                
                # Expand bounds slightly to give room if dataset is very small
                range_span = val_max - val_min
                min_bound = max(0, val_min - (0.1 * range_span if not is_int else 0))
                max_bound = val_max + (0.1 * range_span if not is_int else 0)
                
                if is_int:
                    min_bound = int(np.floor(min_bound))
                    max_bound = int(np.ceil(max_bound))
                    val_mean = int(round(val_mean))
                else:
                    min_bound = float(min_bound)
                    max_bound = float(max_bound)
                    val_mean = float(val_mean)
                
                # Fallback check if min == max
                if min_bound == max_bound:
                    max_bound += 10
                
                input_data[feat] = st.slider(
                    f"{label}",
                    min_value=min_bound,
                    max_value=max_bound,
                    value=val_mean,
                    step=step,
                    key=f"input_{feat}"
                )
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        predict_button = st.button("🔥 Run AI Prediction")
        
    with col_result:
        st.markdown("<h3 style='margin-bottom: 1rem; color: #a855f7;'>📊 Model Output</h3>", unsafe_allow_html=True)
        
        if predict_button:
            # Create a 1-row DataFrame matching training format
            df_input = pd.DataFrame([input_data])
            
            # Apply categorical encoding using saved preprocessor
            categorical_cols = [f for f, info in details.items() if info['type'] == 'categorical']
            if len(categorical_cols) > 0:
                # Apply encoding using same classes
                for col in categorical_cols:
                    le = preprocessor.label_encoders[col]
                    # Map the string input category to integer value
                    try:
                        df_input[col] = le.transform(df_input[col].astype(str))
                    except ValueError:
                        # Fallback for unseen values (although options are restricted)
                        df_input[col] = 0
            
            # Apply standardization scaling using saved scaler
            # Our scaler expects standard scaler object method
            df_scaled = scaler.normalize_scale(df_input, df_input.columns, method='standard', fit=False)
            
            # Perform prediction
            pred_class = model.predict(df_scaled)[0]
            
            # Try to get prediction probability for confidence meter
            has_proba = hasattr(model, 'predict_proba')
            prob_percent = 0.0
            
            if has_proba:
                probs = model.predict_proba(df_scaled)[0]
                prob_percent = probs[pred_class] * 100.0
            
            # Display highly polished result card
            if pred_class == 1:
                st.markdown(f"""
                <div class="result-card-positive">
                    <span style='font-size: 3rem;'>✨</span>
                    <h2 style='background: linear-gradient(135deg, #10b981 0%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0;'>
                        YES / SUCCESSFUL (1)
                    </h2>
                    <p style='color: #a7f3d0; font-size: 1.1rem; margin: 0 0 1.5rem 0;'>
                        Model predicts a positive target result.
                    </p>
                    {"<div style='font-size: 0.95rem; color: #94a3b8;'>Model Confidence: <span style='color: #10b981; font-weight: 700; font-size: 1.3rem;'>" + f"{prob_percent:.2f}%" + "</span></div>" if has_proba else ""}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-negative">
                    <span style='font-size: 3rem;'>✖</span>
                    <h2 style='background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0;'>
                        NO / NEGATIVE (0)
                    </h2>
                    <p style='color: #fca5a5; font-size: 1.1rem; margin: 0 0 1.5rem 0;'>
                        Model predicts a negative target result.
                    </p>
                    {"<div style='font-size: 0.95rem; color: #94a3b8;'>Model Confidence: <span style='color: #ef4444; font-weight: 700; font-size: 1.3rem;'>" + f"{prob_percent:.2f}%" + "</span></div>" if has_proba else ""}
                </div>
                """, unsafe_allow_html=True)
            
            # Display detailed input/output JSON summary inside glass container
            st.markdown("<div class='glass-card' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<div style='color: #a855f7; font-weight: 600; margin-bottom: 0.5rem;'>📋 Data Summary Sent to Model</div>", unsafe_allow_html=True)
            st.json({
                "raw_inputs": input_data,
                "preprocessed_features": df_input.iloc[0].to_dict(),
                "scaled_features": df_scaled.iloc[0].to_dict(),
                "prediction": int(pred_class)
            })
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            # Default helper cards
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem 1.5rem;">
                <span style="font-size: 3rem; opacity: 0.6;">💡</span>
                <h4 style="color: #94a3b8; margin-top: 1rem;">Waiting for Inference</h4>
                <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
                    Adjust the parameters on the left and click the <b>"Run AI Prediction"</b> button to perform model predictions.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar details
    with st.sidebar:
        st.markdown("<h3 style='color: #38bdf8;'>🛠️ System Details</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; font-size: 0.9rem;">
            <b>Trained Model:</b><br>
            <code style="color: #cbd5e1;">{type(model).__name__}</code><br><br>
            <b>Model File:</b><br>
            <code style="color: #cbd5e1;">best_model.pkl</code><br><br>
            <b>Target Label:</b><br>
            <code style="color: #cbd5e1;">{target_name}</code><br><br>
            <b>Number of Features:</b><br>
            <code style="color: #cbd5e1;">{len(features_list)}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center; color: #475569; font-size: 0.8rem; margin-top: 2rem;'>© DASPRO Lab Team 2025</div>", unsafe_allow_html=True)
else:
    st.info("Please resolve the error above to launch the application dashboard.")
