import streamlit as st
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Hardened Mobile-Friendly CSS Engine ---
st.markdown(
    """
<style>
:root {
  --bg: #0A0C14;
  --card: #131726; /* Changed to solid hex to prevent mobile color-inversion washouts */
  --text: #F8FAFC;
  --muted: #94A3B8;
  --border: rgba(139, 92, 246, 0.25);
  
  --violet: #8B5CF6;
  --fuchsia: #D946EF;
  --grad: linear-gradient(135deg, #8B5CF6, #D946EF);
  
  --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
  --radius: 24px;
}

/* Global Overrides */
html, body, p, [class*="css"] {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--text) !important;
}

/* Select / input dropdown POPUP menus must stay readable on DESKTOP.
   Without this, the options list renders on a light background with
   forced-white text (invisible). Guarantee dark bg + light text. */
[data-baseweb="popover"], [data-baseweb="popover"] ul,
[data-baseweb="menu"], [data-baseweb="menu"] ul {
  background-color: var(--card) !important;
}
[data-baseweb="popover"] [role="option"], [data-baseweb="popover"] li,
[data-baseweb="menu"] [role="option"], [data-baseweb="menu"] li {
  background-color: transparent !important;
  color: var(--text) !important;
}
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] li:hover,
[data-baseweb="menu"] [role="option"]:hover, [data-baseweb="menu"] li:hover {
  background-color: rgba(139, 92, 246, 0.35) !important;
  color: #ffffff !important;
}

/* Visible placeholders inside inputs */
input::placeholder, textarea::placeholder {
  color: var(--muted) !important;
  opacity: 1;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
  background-color: var(--bg) !important;
}

.block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 4rem !important;
  max-width: 860px;
}

/* Header Container */
.hero-wrap {
  background: #131726;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 2.5rem;
  margin-bottom: 1.75rem;
  box-shadow: var(--shadow);
}

.hero-grid {
  display: grid;
  grid-template-columns: 70px 1fr;
  gap: 20px;
  align-items: center;
}

.hero-icon {
  width: 64px;
  height: 64px;
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  background: var(--grad);
  display: flex;
  align-items: center;
  justify-content: center;
}

.hero-title {
  font-size: 2.4rem;
  font-weight: 900;
  line-height: 1.1;
  margin: 0;
  letter-spacing: -0.03em;
  background: var(--grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-subtitle {
  margin: 0.5rem 0 0 0;
  color: var(--muted) !important;
  font-size: 1.1rem;
  font-weight: 500;
}

/* Main Form Elements Setup */
.ds-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius) var(--radius) 0 0;
  padding: 2rem 2rem 0.5rem 2rem;
}
.ds-card h3 {
  color: var(--text) !important;
  font-weight: 800;
  margin: 0;
  font-size: 1.5rem;
}
.helper {
  color: var(--muted) !important;
  font-size: 0.95rem;
  margin-top: 0.4rem;
}

/* Grid Layout Matrix Configuration */
div[data-testid="stHorizontalBlock"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 0 var(--radius) var(--radius) !important;
  padding: 2rem !important;
  margin-top: -1px !important; 
  margin-bottom: 2rem !important;
}

/* High Contrast Input Fields for Mobile Forms */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
  background-color: #0A0C14 !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] span, input {
  color: var(--text) !important;
}

[data-testid="stWidgetLabel"] p, strong {
  color: #E2E8F0 !important;
  font-weight: 600 !important;
}

/* Submit Action Button */
.stButton > button, .stFormSubmitButton > button {
  width: 100%;
  border: 0 !important;
  border-radius: 16px !important;
  padding: 1rem 1.5rem !important;
  font-weight: 800 !important;
  text-transform: uppercase;
  background: var(--grad) !important;
  color: #0A0C14 !important;
  box-shadow: 0 15px 35px rgba(139, 92, 246, 0.3) !important;
}

/* Result Banner Layout */
.result-card {
  margin-top: 2rem;
  border-radius: var(--radius);
  border: 1px solid rgba(217, 46, 239, 0.4);
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(217, 46, 239, 0.1));
  padding: 2rem;
}

.result-label {
  margin: 0;
  color: #94A3B8 !important;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  font-size: 0.85rem;
}

.result-value {
  margin: 0.4rem 0 0 0;
  font-size: 4.2rem;
  font-weight: 950;
  background: var(--grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Icon Vector Graphics ---
cap_svg = """
<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 3L1.5 8.25L12 13.5L22.5 8.25L12 3Z" fill="#0A0C14"/>
  <path d="M5.5 10.2V15.2C5.5 16.1 9 18.5 12 18.5C15 18.5 18.5 16.1 18.5 15.2V10.2L12 13.5L5.5 10.2Z" fill="rgba(10,12,20,0.75)"/>
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

# --- Interactive Workspace ---
with st.form("prediction_form"):
    st.markdown(
        """
<div class="ds-card">
  <h3>Student Profile Information</h3>
  <div class="helper">Input metrics directly to calculate validation matrices.</div>
</div>
""",
        unsafe_allow_html=True,
    )

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

    submit_button = st.form_submit_button(label="Predict Math Score", use_container_width=True)

# --- Core Pipeline Integration Matrix ---
if submit_button:
    with st.spinner("Processing parameters via prediction matrices..."):
        try:
            # LAZY LOADING IMPORTS: Bypasses mobile load-time race conditions entirely
            from src.pipeline.predict_pipeline import CustomData, PredictPipeline

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

            arr = np.asarray(results).ravel()
            if arr.size == 0:
                raise ValueError(f"Model returned empty predictions: {results}")

            final_score = round(float(arr[0]), 2)

            st.markdown(
                f"""
<div class="result-card">
  <p class="result-label">Predicted Math Score</p>
  <p class="result-value">{final_score} <span style="font-size:1.5rem; font-weight:800; color: var(--fuchsia); -webkit-text-fill-color: var(--fuchsia); opacity:0.8;">/ 100</span></p>
</div>
""",
                unsafe_allow_html=True,
            )

        except FileNotFoundError:
            st.error("🚨 Missing model files: ensure your `artifacts/` tree contains valid serialization components.")
        except Exception as e:
            st.error(f"⚠️ An error occurred during pipeline calculation: {e}")
