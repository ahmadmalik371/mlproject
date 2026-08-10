# 🎓 Student Performance Prediction (ML Project)

An end-to-end **machine learning** project that predicts a student's **math score** from demographic and academic features. It includes a complete training pipeline, model comparison with hyper-parameter tuning, and an interactive **Streamlit** web application with a custom mobile-friendly UI.

## 🚀 Live Demo

Try the deployed app:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-score-predictor-ml1.streamlit.app/)

🔗 **https://student-score-predictor-ml1.streamlit.app/**


---

## 📌 Project Overview

The system estimates a student's final **math score** (0–100) based on:

| Category | Features |
|----------|----------|
| **Demographics** | Gender, Race/Ethnicity, Parental Level of Education |
| **Academic & Support** | Lunch type, Test preparation course |
| **Prior performance** | Reading score, Writing score |

The project follows a classic MLOps-style structure: data ingestion → data transformation → model training → model evaluation → prediction served through a web app.

---

## 🧠 How It Works

1. **Data Ingestion** (`src/components/data_ingestion.py`)
   - Reads the raw dataset from `notebook/stud.csv`.
   - Splits it into train (`artifacts/train.csv`) and test (`artifacts/test.csv`) sets (80/20, `random_state=42`).

2. **Data Transformation** (`src/components/data_transformation.py`)
   - Builds a `ColumnTransformer` preprocessor:
     - **Numerical** (`reading_score`, `writing_score`): median imputation → `StandardScaler`.
     - **Categorical** (the 5 demographic/support features): `most_frequent` imputation → `OneHotEncoder` → `StandardScaler(with_mean=False)`.
   - Saves the fitted preprocessor to `artifacts/preprocessor.pkl`.

3. **Model Training** (`src/components/model_trainer.py`)
   - Evaluates **7 regression models** using `GridSearchCV` (3-fold) for hyper-parameter tuning.
   - Compares them by **R² score** on the test set.
   - Selects and persists the best model to `artifacts/model.pkl`.
   - If the best R² is below `0.6`, training fails with *"No best model found"*.

   **Models compared:**
   - Random Forest
   - Decision Tree
   - Gradient Boosting
   - Linear Regression
   - XGBoost (`XGBRegressor`)
   - CatBoost (`CatBoostRegressor`)
   - AdaBoost (`AdaBoostRegressor`)

4. **Prediction** (`src/pipeline/predict_pipeline.py`)
   - `CustomData` collects a single student profile and converts it to a DataFrame.
   - `PredictPipeline` loads the saved model + preprocessor and returns the predicted math score.

5. **Web App** (`app.py`)
   - A **Streamlit** application that collects student details in a form and displays the predicted math score in a polished, mobile-responsive dark-themed UI.

---

## 📁 Project Structure

```
mlproject/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup (editable install)
├── README.md
├── artifacts/                      # Generated outputs
│   ├── data.csv                    # Raw dataset copy
│   ├── train.csv                   # Training split
│   ├── test.csv                    # Test split
│   ├── preprocessor.pkl            # Saved preprocessing pipeline
│   └── model.pkl                   # Saved best model
├── logs/                           # Application logs (auto-generated)
├── catboost_info/                  # CatBoost training output
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb   # Exploratory data analysis
│   ├── 2. MODEL TRAINING.ipynb             # Experimentation notebook
│   └── stud.csv                            # Raw dataset
└── src/
    ├── __init__.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    ├── pipeline/
    │   ├── __init__.py
    │   ├── predict_pipeline.py
    │   └── train_pipeline.py
    ├── exception.py                # Custom exception handling
    ├── logger.py                   # Logging configuration
    └── utils.py                    # Shared helpers (save/load/evaluate)
```

---

## 🚀 Getting Started

### 1. Clone & set up a virtual environment

```bash
git clone https://github.com/ahmadmalik371/mlproject.git
cd mlproject
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Optionally install the project itself in editable mode: `pip install -e .`
> (edit `setup.py` if you want to change author/project metadata).

### 3. Train the model (optional)

If you want to re-run the ingestion → transformation → training pipeline:

```bash
python -m src.components.data_ingestion
```

This regenerates the artifacts in the `artifacts/` folder.

### 4. Run the web app

```bash
streamlit run app.py
```

Open the printed local URL (default: `http://localhost:8501`), fill in the student profile, and click **Predict Math Score**.

---

## 🧪 Testing the Prediction Pipeline

For a quick scripted prediction without the UI:

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=70,
    writing_score=72,
)

pred_df = data.get_data_as_data_frame()
result = PredictPipeline().predict(pred_df)
print("Predicted math score:", result[0])
```

---

## 🔧 Dependencies

See [`requirements.txt`](requirements.txt):

- `pandas`, `numpy` — data manipulation
- `seaborn`, `matplotlib` — visualization (used in notebooks)
- `scikit-learn` — modeling & preprocessing
- `catboost`, `xgboost` — gradient boosting libraries
- `dill`, `pickle` — object serialization
- `streamlit` — web application framework

---

## 🗂️ Dataset

The raw dataset is the classic **"Students Performance"** dataset stored at `notebook/stud.csv`. It contains ~1000 student records with columns:
`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `reading_score`, `writing_score`, and the target `math_score`.

---

## 📝 Notes & Limitations

- The prediction pipeline expects the saved artifacts (`artifacts/model.pkl` and `artifacts/preprocessor.pkl`). If they are missing, the app shows an error and you must re-train the model.
- The app UI reflects the exact categorical values/options defined in the training data — keep these in sync if you retrain on a different dataset.
- Model performance is measured via **R² score**; tune the `params` grids in `model_trainer.py` to improve accuracy.

---

## 👤 Author

- **Ahmad** — [realahmadmalik3@gmail.com](mailto:realahmadmalik3@gmail.com)

## 📄 License

This project is for educational purposes. Check the underlying dataset's license before commercial use.
