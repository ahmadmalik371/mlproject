import streamlit as st
import numpy as np

# Import your existing pipeline modules
try:
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
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

# --- Custom CSS Styling (Modern Light + Indigo/Purple Accents) ---
st.markdown(
    """
<style>
:root{
  --bg: #F7F8FC;
  --card: rgba(255,255,255,0.86);
  --text: #0F172A;
  --muted: #64748B;
  --border: rgba(15, 23, 42, 0.10);

  --indigo: #4F46E5;
  --purple: #7C3AED;

  --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
  --shadow-2: 0 10px 25px rgba(15, 23, 42, 0.08);
  --radius: 16px;
}

html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  color: var(--text);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
  background: radial-gradient(1200px 600px at 10% 0%, rgba(124, 58, 237, 0.10), transparent 55%),
              radial-gradient(900px 500px at 90% 10%, rgba(79, 70, 229, 0.10), transparent 55%),
              linear-gradient(180deg, var(--bg), #FFFFFF);
}

.block-container {
  padding-top: 2.25rem !important;
  padding-bottom: 3rem !important;
  max-width: 980px;
}

.hero-wrap{
  background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(124,58,237,0.08));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.35rem 1.5rem;
  margin-bottom: 1.25rem;
  position: relative;
  overflow: hidden;
}

.hero-grid{
  position: relative;
  display: grid;
  grid-template-columns: 64px 1fr;
  gap: 14px;
  align-items: center;
}

.hero-icon{
  width: 56px;
  height: 56px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(79,70,229,0.18), rgba(124,58,237,0.16));
  border: 1px solid rgba(79,70,229,0.18);
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow: var(--shadow-2);
}

.hero-title{
  font-size: 2.05rem;
  font-weight: 850;
  line-height: 1.1;
  margin: 0;
  letter-spacing: -0.02em;
}

.hero-subtitle{
  margin: 0.35rem 0 0 0;
  color: var(--muted);
  font-size: 1.02rem;
}

.ds-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.15rem 0.25rem 1.15rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  margin-top: 0.75rem;
}

.helper{
  color: var(--muted);
  font-size: 0.92rem;
  margin-top: -0.25rem;
  margin-bottom: 0.9rem;
}

.stButton > button, .stFormSubmitButton > button{
  width: 100%;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.85rem 1.05rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.01em !important;
  background: linear-gradient(135deg, var(--indigo), var(--purple)) !important;
  color: white !important;
  box-shadow: 0 14px 30px rgba(79,70,229,0.28) !important;
}

.result-card{
  margin-top: 1rem;
  border-radius: var(--radius);
  border: 1px solid rgba(79,70,229,0.18);
  background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(124,58,237,0.08));
  box-shadow: var(--shadow);
  padding: 1.25rem 1.25rem;
}

.result-label{
  margin: 0;
  color: rgba(15,23,42,0.75);
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.82rem;
}

.result-value{
  margin: 0.35rem 0 0 0;
  font-size: 3.0rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  color: rgba(15,23,42,0.92);
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Hero Section ---
cap_svg = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 3L1.5 8.25L12 13.5L22.5 8.25L12 3Z" fill="rgba(79,70,229,0.95)"/>
  <path d="M5.5 10.2V15.2C5.5 16.1 9 18.5 12 18.5C15 18.5 18.5 16.1 18.5 15.2V10.2L12 13.5L5.5 10.2Z" fill="rgba(124,58,237,0.85)"/>
  <path d="M22.5 8.25V14.25" stroke="rgba(15,23,42,0.55)" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M22.5 14.25C21.7 14.25 21 14.95 21 15.75C21 16.55 21.7 17.25 22.5 17.25C23.3 17.25 24 16.55 24 15.75C24 14.95 23.3 14.25 22.5 14.25Z" fill="rgba(15,23,42,0.35)"/>
</svg>
"""

st.markdown(
    f"""
<div class="hero-wrap">
  <div class="hero-grid">
    <div class="hero-icon">{cap_svg}</div>
    <div>
      <p class="hero-title">Student Math Score Predictor</p>
      <p class="hero-subtitle">Estimate math performance from demographics and academic context.</p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- Main Application Layout ---
with st.form("prediction_form"):
    st.markdown(
        """
<div class="ds-card">
  <h3>Student Profile Information</h3>
  <div class="helper">Fill in the details below.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ds-card">', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", options=["male", "female"])
        race_ethnicity = st.selectbox("Race or Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            options=[
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ],
        )

    with right:
        st.markdown("**Academics & Support**")
        lunch = st.selectbox("Lunch Type", options=["standard", "free/reduced"])
        test_preparation_course = st.selectbox("Test Preparation Course", options=["none", "completed"])

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50, step=1)
        with c2:
            writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50, step=1)

    st.markdown("</div>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Predict Math Score", use_container_width=True)

# --- Prediction Logic (FIXED) ---
if submit_button:
    with st.spinner("Analyzing profile and generating prediction..."):
        try:
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score,
            )

            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # IMPORTANT FIX: extract a single scalar from array/list outputs
            final_score = np.asarray(results).reshape(-1)
            final_score = round(float(final_score), 2)

            st.markdown(
                f"""
<div class="result-card">
  <p class="result-label">Predicted Math Score</p>
  <p class="result-value">{final_score} <span style="font-size:1.15rem; font-weight:800; color: rgba(15,23,42,0.55);">/ 100</span></p>
</div>
""",
                unsafe_allow_html=True,
            )

            if final_score > 100:
                st.info("Note: Depending on the model, predictions can occasionally exceed 100 slightly.")

        except FileNotFoundError:
            st.error("🚨 Missing model files: ensure `artifacts/` contains `model.pkl` and `preprocessor.pkl`.")
        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")
