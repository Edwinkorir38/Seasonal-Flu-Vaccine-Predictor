# 🧪 Seasonal Flu Vaccine Uptake Predictor  
![Project Banner](Images/influenza-vaccine1.png)

<p align="center">
  <a href="https://seasonal-flu-vaccine-predictor-pbptcx6ejexsetogl6udl3.streamlit.app/">
    <img src="https://img.shields.io/badge/🌐_Live_App-Visit-brightgreen?style=for-the-badge">
  </a>
  <a href="https://github.com/Edwinkorir38/Seasonal-Flu-Vaccine-Predictor">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-success?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge">
</p>

---

## 👤 **Author**
**[Edwin Korir](https://github.com/Edwinkorir38)**  
📩 **LinkedIn:** https://www.linkedin.com/in/edwin-korir-90a794382  

---

## 📌 **Project Overview**

This project predicts whether an individual is likely to receive the **seasonal flu vaccine**, using survey data collected during the 2009 H1N1 pandemic.  

The objective is to help:

- 🏥 **Public health officials**  
- 👩‍⚕️ **Healthcare providers**  
- 🔬 **Data scientists & epidemiologists**  

… understand key factors behind vaccine acceptance, enabling **better outreach, messaging, and targeted health interventions**.

---

## 📂 **Repository Structure**
```
📁 Seasonal-Flu-Vaccine-Predictor/
│── app.py # Streamlit application
│── train_model.py # Model training script
│── requirements.txt
│── defaults.pkl
│── feature_list.pkl
│── full_feature_list.pkl
│── seasonal_flu_pipeline.pkl # Trained ML pipeline
│── H1N1_and_Seasonal_Flu_Vaccines.ipynb
│── Images/ # Plots, charts & visuals
└── README.md
```

---

# 📊 **1. Exploratory Data Analysis (EDA)**

### 🎯 **Target Distribution**
Most respondents **did not receive** the seasonal flu vaccine.

![Distribution](Images/seasonal-vaccine-count-plot.png)

---

### 🔗 **Feature Correlation Map**

Key positive correlates:
- `doctor_recc_seasonal`
- `opinion_seas_risk`
- `opinion_seas_vacc_effective`

![Correlation Map](Images/corr-map.png)

---

### 🧩 **Missing Data Overview**

![Missing Values](Images/missing-data-in-Train-dataset.png)

---

# 🤖 **2. Modeling Approach**

### 🔧 **Preprocessing Steps**
- Missing value handling (median/mode)
- Label encoding for categorical features
- Train-test stratified split
- Mutual information + model-based feature importance

### 🧪 **Models Evaluated**
- Logistic Regression  
- Decision Tree  
- Random Forest ⭐ **Best performance**  
- XGBoost  

### 🌟 **Top Feature Importance (Random Forest)**  
![Feature Importance](Images/random-forest-feature-importance.png)

---

# 📈 **3. Model Evaluation**

| Model              | Accuracy | Recall | Precision | Train AUC | Test AUC |
|-------------------|----------|--------|-----------|-----------|----------|
| Logistic Regression | 78.2% | 73.9% | 77.4% | 85.0% | 85.2% |
| Decision Tree       | 75.8% | 67.5% | 76.9% | 83.1% | 82.6% |
| Random Forest       | **78.4%** | 72.9% | 78.3% | **90.4%** | **85.4%** |
| XGBoost             | 77.0% | 74.0% | 75.3% | 87.5% | 76.7% |

### 🧭 **ROC Curve Comparison**
![ROC Comparison](Images/all-roc-curve.png)

---

# 📝 **4. Conclusions**

### 🎉 **Key Insights**
- Doctor recommendations are the **strongest predictor**.  
- Vaccine **risk & effectiveness perceptions** heavily influence uptake.  
- Older age groups are significantly more likely to vaccinate.  

### ⚖️ **Model Summary**
The **Random Forest** model performed best with a test AUC of **0.8539**.

---

# 📌 **5. Recommendations**

### ✔ Public Health Actions  
- Strengthen **doctor-driven communication**  
- Target **younger demographics**  
- Improve messaging around vaccine **safety & effectiveness**

### ✔ Technical Improvements  
- Use **SMOTE / class rebalancing techniques**  
- Add **LIME / SHAP explainability**  
- Fine-tune with more recent **post-COVID** data

---

# 🚀 **6. Deployment**

This project is deployed using **Streamlit Cloud**.

### 👉 **Live App (Click to Open):**  
https://seasonal-flu-vaccine-predictor-pbptcx6ejexsetogl6udl3.streamlit.app/

### Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
