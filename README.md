# 🦴 Predicting Osteoporosis Risk
### Machine Learning for Early Detection & Clinical Decision Support
**Author:** Afeef Anver sha &nbsp;|&nbsp; **Organization:** First Quadrant Labs &nbsp;|&nbsp; **Program:** BIA®

---

## 📌 Project Overview

Osteoporosis is a silent disease affecting **1 in 3 women** and **1 in 5 men** over the age of 50, often going undetected until a fracture occurs. This project builds a comprehensive end-to-end machine learning pipeline to predict osteoporosis risk from patient demographics, hormonal status, nutritional habits, lifestyle factors, and medical history.

### Goals
- **Early Detection** — Identify high-risk patients before fractures occur
- **Personalised Care** — Support clinicians with data-driven risk profiles
- **Model Transparency** — Explain predictions using permutation importance (SHAP-style)
- **Comparative Analysis** — Evaluate 9 ML models systematically

---

## 📁 Repository Structure

```
osteoporosis-risk-prediction/
│
├── NB1_EDA.ipynb                    # Exploratory Data Analysis
├── NB2_Model_Training_v4.ipynb      # Preprocessing, Baseline & GridSearchCV (9 Models)
├── NB3_Model_Evaluation_v3.ipynb    # Full Evaluation & Comparative Analysis
├── osteoporosis_web_app_v3.html     # Interactive 9-Model Web App
├── Osteoporosis_Risk_Prediction.pptx # Project Presentation (11 slides)
├── osteoporosis.csv                 # Dataset (place here before running)
├── models/                          # Saved model artifacts (generated on run)
│   ├── *.pkl                        # Trained models & scaler
│   ├── feature_names.pkl
│   ├── baseline_results.csv
│   ├── tuned_results.csv
│   └── final_metrics.csv
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Features | 16 (14 input + 1 ID + 1 target) |
| Target | `Osteoporosis` (Yes / No) |
| Feature Types | Numerical, Binary, Ordinal, Nominal |

### Feature Categories

| Category | Features |
|----------|----------|
| Demographics | Age, Gender, Race/Ethnicity |
| Hormonal & Genetic | Hormonal Changes, Family History |
| Nutrition | Calcium Intake, Vitamin D Intake |
| Lifestyle | Physical Activity, Smoking, Alcohol Consumption |
| Medical | Medical Conditions, Medications, Prior Fractures, Body Weight |

---

## ⚙️ Methodology

The project follows the **correct ML workflow** — baseline training before any tuning:

```
Raw Data
   │
   ├─► Data Cleaning & Imputation
   │
   ├─► Train / Test Split  ◄── Split FIRST (before encoding/scaling)
   │
   ├─► Feature Encoding (rule-based, no target stats)
   │
   ├─► Feature Engineering
   │       • Nutrient_Deficiency
   │       • Lifestyle_Risk
   │       • Age_Group
   │       • Hormonal_Bone_Risk
   │
   ├─► StandardScaler  ◄── Fit on TRAIN only
   │
   ├─► Baseline Training (default params, all 9 models)
   │
   ├─► Baseline Analysis (CV F1, overfitting check)
   │
   ├─► GridSearchCV (5-fold StratifiedKFold, train set only)
   │
   ├─► Tuned vs Baseline Comparison
   │
   └─► Final Test Evaluation (NB3)
```

### 🔒 Data Leakage Safeguards

| Risk | Safeguard |
|------|-----------|
| Scaler fit on full data | `StandardScaler` fit **only on X_train** |
| Encoding uses target | All mappings are domain constants — no target stats |
| CV leaks test data | `StratifiedKFold` runs on **train set only** |
| GridSearch before baseline | Baseline trained first — correct workflow enforced |
| Test set seen early | X_test sealed — only used in NB3 |

---

## 🤖 Models Used

| # | Model | Type | Key Params Tuned |
|---|-------|------|-----------------|
| 1 | Logistic Regression | Linear | `C`, `solver` |
| 2 | Naive Bayes | Probabilistic | `var_smoothing` |
| 3 | KNN | Instance-based | `n_neighbors`, `metric` |
| 4 | Decision Tree | Tree | `max_depth`, `criterion`, `min_samples_leaf` |
| 5 | Random Forest | Bagging | `n_estimators`, `max_depth`, `max_features` |
| 6 | Extra Trees | Bagging | `n_estimators`, `max_depth`, `max_features` |
| 7 | AdaBoost | Boosting | `n_estimators`, `learning_rate` |
| 8 | Gradient Boosting | Boosting | `learning_rate`, `max_depth` |
| 9 | XGBoost | Boosting | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

---

## 📈 Results

### Test Set Performance — All 9 Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.803 | 0.799 | 0.807 | 0.803 | 0.871 |
| Naive Bayes | 0.771 | 0.758 | 0.781 | 0.769 | 0.848 |
| KNN | 0.788 | 0.781 | 0.796 | 0.788 | 0.855 |
| Decision Tree | 0.832 | 0.828 | 0.837 | 0.832 | 0.887 |
| Random Forest | 0.879 | 0.873 | 0.885 | 0.879 | 0.934 |
| Extra Trees | 0.868 | 0.862 | 0.874 | 0.868 | 0.922 |
| AdaBoost | 0.841 | 0.836 | 0.847 | 0.841 | 0.899 |
| Gradient Boosting | 0.889 | 0.883 | 0.895 | 0.889 | 0.941 |
| **XGBoost 🏆** | **0.901** | **0.896** | **0.907** | **0.901** | **0.953** |

### 🏆 Top 3 Models

| Rank | Model | F1 Score | ROC-AUC |
|------|-------|----------|---------|
| 🥇 1st | XGBoost | 0.901 | 0.953 |
| 🥈 2nd | Gradient Boosting | 0.889 | 0.941 |
| 🥉 3rd | Random Forest | 0.879 | 0.934 |

---

## 🔍 Key Clinical Findings

### Top Risk Factors (Permutation Importance — averaged across all 9 models)

| Rank | Feature | Avg F1 Drop |
|------|---------|-------------|
| 1 | Prior Fractures | 18.2% |
| 2 | Hormonal Changes | 16.1% |
| 3 | Family History | 13.8% |
| 4 | Age | 12.4% |
| 5 | Calcium Intake | 10.9% |
| 6 | Vitamin D Intake | 9.7% |
| 7 | Gender | 8.5% |
| 8 | Physical Activity | 7.4% |

### Clinical Insights
- 🔴 Postmenopausal women with family history face **3× higher risk**
- 🔴 Prior fractures are the **strongest single predictor** (18.2% F1 drop)
- 🟡 Low calcium + insufficient vitamin D **compound risk significantly**
- 🟡 Corticosteroid use independently **doubles bone loss risk**
- 🟢 Active lifestyle reduces osteoporosis rate by **~40%** vs sedentary

---

## 🧪 Sample Test Cases

| ID | Age | Sex | Hormonal | Family | Calcium | Vit D | Activity | Smoking | Prior Fx | **Risk** |
|----|-----|-----|----------|--------|---------|-------|----------|---------|----------|----------|
| P01 | 72 | F | Postmeno. | Yes | Low | Insuff. | Sedentary | Yes | Yes | 🔴 **High** |
| P02 | 65 | F | Postmeno. | Yes | Low | Insuff. | Moderate | No | Yes | 🔴 **High** |
| P03 | 58 | F | Postmeno. | No | Adequate | Insuff. | Sedentary | Yes | No | 🔴 **High** |
| P04 | 55 | M | Normal | Yes | Low | Insuff. | Moderate | No | Yes | 🟡 **Moderate** |
| P05 | 48 | F | Normal | No | Adequate | Insuff. | Sedentary | Yes | No | 🟡 **Moderate** |
| P06 | 38 | M | Normal | No | Adequate | Suff. | Active | No | No | 🟢 **Low** |
| P07 | 42 | F | Normal | No | Adequate | Suff. | Active | No | No | 🟢 **Low** |

---

## 🌐 Web App

An interactive web app (`osteoporosis_web_app_v3.html`) allows real-time risk assessment across all 9 models.

### Features
- **Model Selector** — Switch between all 9 models via chip buttons
- **All 9 Results Table** — Ranked by risk %, with 🥇🥈🥉 medals
- **Top 3 Podium** — Best models highlighted with "Use This" button
- **9-Model Consensus** — Agreement %, average score, dominant verdict
- **SHAP-style Impact Bars** — Feature contribution for each model
- **Clinical Advice Tab** — Tailored recommendations for Low / Moderate / High risk

> Open `osteoporosis_web_app_v3.html` directly in any browser — no server required.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Running the Notebooks

Run in order:

```bash
jupyter notebook NB1_EDA.ipynb
jupyter notebook NB2_Model_Training_v4.ipynb
jupyter notebook NB3_Model_Evaluation_v3.ipynb
```

> Make sure `osteoporosis.csv` is in the same directory before running.

### Running the Web App

Simply open in your browser:

```bash
open osteoporosis_web_app_v3.html   # macOS
start osteoporosis_web_app_v3.html  # Windows
```

---

## 🔮 Future Work

- [ ] Walk-forward validation & backtesting framework
- [ ] Bayesian hyperparameter optimisation (Optuna)
- [ ] Model drift detection for production deployment
- [ ] Integrate DEXA scan imaging data as additional features
- [ ] Deploy as a Streamlit or FastAPI web service

---

## 📄 License

This project was developed as part of the **BIA® Research Program** at **First Quadrant Labs**.  
For academic and research purposes only.

---

## 🙏 Acknowledgements

- **First Quadrant Labs** — for the project brief and mentorship
- **BIA® Program** — for the structured learning framework
- **Scikit-learn, XGBoost** — core ML libraries used throughout

---

> ⚕️ **Clinical Disclaimer:** This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis or clinical advice.
