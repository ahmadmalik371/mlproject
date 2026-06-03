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

# --- Midnight Obsidian CSS Framework ---
st.markdown(
    """
<style>
:root{
  --bg: #070A12;
  --card: rgba(15, 23, 42, 0.65);
  --text: #F1F5F9;
  --muted: #64748B;
  --border: rgba(99, 102, 241, 0.15);

  --indigo: #6366F1;
  --teal: #14B8A6;
  --grad: linear-gradient(135deg, #6366F1, #14B8A6);
  --grad-glow: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(20, 184, 166, 0.1));

  --shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
  --shadow-hover: 0 30px 60px rgba(20, 184, 166, 0.15);
  --radius: 20px;
}

/* Force dark theme overrides and stop mobile inversion bugs */
html, body, p, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
  color: var(--text) !important;
}

#MainMenu, footer, header {visibility: hidden;}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(15px); }
  to { opacity: 1; transform: translateY(0); }
}

.stApp {
  background: radial-gradient(1000px 600px at 0% 0%, rgba(99, 102, 241, 0.12), transparent 50%),
              radial-gradient(800px 600px at 100% 100%, rgba(20, 184, 166, 0.12), transparent 50%),
              var(--bg);
  animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.block-container {
  padding-top: 2rem !important;
  padding-bottom: 3rem !important;
  max-width: 900px;
}

/* Premium Obsidian Hero Banner */
.hero-wrap {
  position: relative;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.5));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 2.2rem 2.2rem 4.5rem 2.2rem;
  margin-bottom: 1.5rem;
  overflow: hidden;
  box-shadow: var(--shadow);
}

.hero-grid {
  display: grid;
  grid-template-columns: 68px 1fr;
  gap: 18px;
  align-items: center;
  position: relative;
  z-index: 5;
}

/* Futuristic Hex/Polygon Custom Clip-Path Selector */
.hero-icon {
  width: 60px;
  height: 60px;
  clip-path: polygon(25% 5%, 75% 5%, 100% 30%, 100% 70%, 75% 95%, 25% 95%, 0% 70%, 0% 30%);
  background: var(--grad);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 10px 25px rgba(20, 184, 166, 0.3);
  transition: transform 0.4s ease;
}
.hero-wrap:hover .hero-icon {
  transform: rotate(15deg) scale(1.08);
}

/* Neon Gradient Text Blending */
.hero-title {
  font-size: 2.25rem;
  font-weight: 900;
  line-height: 1.1;
  margin: 0;
  letter-spacing: -0.03em;
  background: var(--grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-subtitle {
  margin: 0.4rem 0 0 0;
  color: var(--muted) !important;
  font-size: 1.05rem;
  font-weight: 500;
}

/* Card Header Module */
.ds-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius) var(--radius) 0 0;
  padding: 1.75rem 1.75rem 0.5rem 1.75rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
}
.ds-card h3 {
  color: var(--text) !important;
  font-weight: 850;
  letter-spacing: -0.01em;
  margin: 0;
}
.helper {
  color: var(--muted) !important;
  font-size: 0.95rem;
  margin-top: 0.35rem;
}

/* Glassmorphism Layout Engine Grid */
div[data-testid="stHorizontalBlock"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 0 var(--radius) var(--radius) !important;
  padding: 1.8rem !important;
  box-shadow: var(--shadow) !important;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  margin-top: -1px !important; 
  margin-bottom: 1.5rem !important;
  transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.4s ease, border-color 0.4s ease !important;
}

div[data-testid="stHorizontalBlock"]:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-hover) !important;
  border-color: rgba(20, 184, 166, 0.3) !important;
}

/* Input Form Widgets Style Integrations */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
  background-color: rgba(15, 23, 42, 0.6) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  border-radius: 10px !important;
}
div[data-baseweb="select"] span, input {
  color: var(--text) !important;
}

[data-testid="stWidgetLabel"] p, strong {
  color: #94A3B8 !important;
  font-weight: 600 !important;
  font-size: 0.95rem;
}

/* Transform State Functions on Neon Buttons */
.stButton > button, .stFormSubmitButton > button {
  width: 100%;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.95rem 1.2rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  background: var(--grad) !important;
  color: #070A12 !important; /* High contrast dark text inside neon button */
  box-shadow: 0 12px 28px rgba(20, 184, 166, 0.25) !important;
  transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease !important;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 18px 36px rgba(20, 184, 166, 0.45) !important;
}

/* Cybernetic Neon Display Card */
.result-card {
  margin-top: 1.5rem;
  border-radius: var(--radius);
  border: 1px solid rgba(20, 184, 166, 0.3);
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(20, 184, 166, 0.05));
  box-shadow: var(--shadow-hover);
  padding: 1.75rem;
  position: relative;
  overflow: hidden;
}

.result-label {
  margin: 0;
  color: #94A3B8 !important;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.85rem;
}

.result-value {
  margin: 0.25rem 0 0 0;
  font-size: 3.8rem;
  font-weight: 950;
  letter-spacing: -0.04em;
  background: var(--grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Vector Scaling Header Waves (Updated Cyber Colors) ---
cap_svg = """
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 3L1.5 8.25L12 13.5L22.5 8.25L12 3Z" fill="#070A12"/>
  <path d="M5.5 10.2V15.2C5.5 16.1 9 18.5 12 18.5C15 18.5 18.5 16.1 18.5 15.2V10.2L12 13.5L5.5 10.2Z" fill="rgba(7,10,18,0.75)"/>
</svg>
"""

organic_waves = """
<svg viewBox="0 0 1440 160" style="position:absolute; bottom:0; left:0; width:100%; height:auto; z-index:1; pointer-events:none; min-width:900px;">
  <path fill="rgba(20, 184, 166, 0.06)" d="M0,96L60,101.3C120,107,240,117,360,112C480,107,600,85,720,80C840,75,960,85,1080,90.7C1200,96,1320,96,1380,96L1440,96L1440,160L1380,160C1320,160,1200,160,1080,160C960,160,840,160,720,160C600,160,480,160,360,160C240,160,120,160,60,160L0,160Z"></path>
  <path fill="rgba(99, 102, 241, 0.04)" d="M0,128L60,122.7C120,117,240,107,360,96C480,85,600,75,720,80C840,85,960,107,1080,112C1200,117,1320,107,1380,101.3L1440,96L1440,160L1380,160C1320,160,1200,160,1080,160C960,160,840,160,720,160C600,160,480,160,360,160C240,160,120,160,60,160L0,160Z"></path>
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
  {organic_waves}
</div>
""",
    unsafe_allow_html=True,
)

# --- Form Block Layout ---
with st.form("prediction_form"):
    st.markdown(
        """
<div class="ds-card">
  <h3>Student Profile Information</h3>
  <div class="helper">Input demographic parameters to deploy predictive metrics.</div>
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

# --- Optimized Core Prediction Logic ---
if submit_button:
    with st.spinner("Processing parameters via pipeline matrices..."):
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

            arr = np.asarray(results).ravel()
            if arr.size == 0:
                raise ValueError(f"Model returned empty predictions: {results}")

            # Precise extraction of the 0-dimensional scalar float
            final_score = round(float(arr[0]), 2)

            st.markdown(
                f"""
<div class="result-card">
  <p class="result-label">Predicted Math Score</p>
  <p class="result-value">{final_score} <span style="font-size:1.4rem; font-weight:800; color: var(--teal); -webkit-text-fill-color: var(--teal); opacity:0.8;">/ 100</span></p>
</div>
""",
                unsafe_allow_html=True,
            )

            if final_score > 100:
                st.info("Note: System calculations can occasionally scale above a baseline of 100.")

        except FileNotFoundError:
            st.error("🚨 Missing model files: ensure your `artifacts/` tree contains valid serialization components.")
        except Exception as e:
            st.error(f"⚠️ An error occurred during pipeline calculation: {e}")
