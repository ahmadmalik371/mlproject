import streamlit as st
import pandas as pd
import sys

# Import your existing pipeline modules
try:
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
    from src.exception import CustomException
except ModuleNotFoundError:
    st.error("⚠️ Could not locate the `src` module. Ensure `app.py` is in the root directory alongside `src/`.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling ---
# Injects custom styling to modernize the UI, hide default Streamlit menus, and style the results card.
st.markdown("""
    <style>
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section Styling */
    .hero-title {
        font-size: 3rem !important;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0rem;
        padding-top: 2rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Result Card Styling */
    .result-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .result-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
    }
    .result-label {
        font-size: 1.2rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown('<p class="hero-title">🎓 Student Math Score Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Leverage Machine Learning to estimate student performance based on demographic and academic background.</p>', unsafe_allow_html=True)

# --- Main Application Layout ---
with st.form("prediction_form"):
    st.subheader("Student Profile Information")
    
    # Using columns for a cleaner, compact form layout
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["male", "female"])
        race_ethnicity = st.selectbox("Race or Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox(
            "Parental Level of Education", 
            options=["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
        )
        
    with col2:
        lunch = st.selectbox("Lunch Type", options=["standard", "free/reduced"])
        test_preparation_course = st.selectbox("Test Preparation Course", options=["none", "completed"])
        
        # Input validation: Scores must be between 0 and 100
        reading_score = st.number_input("Reading Score (0-100)", min_value=0, max_value=100, value=50, step=1)
        writing_score = st.number_input("Writing Score (0-100)", min_value=0, max_value=100, value=50, step=1)

    # Submission button
    submit_button = st.form_submit_button(label="Predict Math Score", use_container_width=True)

# --- Prediction Logic ---
if submit_button:
    with st.spinner("Analyzing profile and generating prediction..."):
        try:
            # 1. Map UI inputs to the CustomData structure expected by your pipeline
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )
            
            # 2. Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            
            # 3. Initialize pipeline and predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Extract and format the result (rounding the floating point)
            final_score = round(results[0], 2)
            
            # 4. Display modern results panel
            st.markdown(f"""
                <div class="result-card">
                    <p class="result-label">Predicted Math Score</p>
                    <p class="result-value">{final_score} / 100</p>
                </div>
            """, unsafe_allow_html=True)
            
            if final_score > 100:
                st.info("Note: Depending on the model's regression bounds, predictions can occasionally slightly exceed 100.")
                
        except FileNotFoundError:
            st.error("🚨 **Missing Model Files:** The application could not find `model.pkl` or `preprocessor.pkl`. Ensure you have run your training pipeline and the `artifacts/` folder contains these files.")
        except Exception as e:
            st.error(f"⚠️ **An error occurred during prediction:** {str(e)}")