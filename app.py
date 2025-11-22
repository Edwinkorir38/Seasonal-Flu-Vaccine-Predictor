import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Seasonal Flu Vaccine Predictor",
    layout="centered",
    page_icon="💉"
)

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
MODEL_PATH = "seasonal_flu_pipeline.pkl"
FULL_FEATURES_PATH = "full_feature_list.pkl"
TOP8_PATH = "feature_list.pkl"
DEFAULTS_PATH = "defaults.pkl"
TRAIN_FEATURES_CSV = "Data/training_set_features.csv"

model = joblib.load(MODEL_PATH)
full_feature_list = joblib.load(FULL_FEATURES_PATH)
top8 = joblib.load(TOP8_PATH)
defaults = joblib.load(DEFAULTS_PATH)

try:
    train_df = pd.read_csv(TRAIN_FEATURES_CSV)
except:
    train_df = None


# ---------------------------------------------------------
# STYLE (Custom CSS for Modern UI)
# ---------------------------------------------------------
st.markdown("""
<style>
body {font-family: 'Helvetica', sans-serif;}

.section-card {
    background: #f8f9fb;
    padding: 1.5rem 2rem;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    border-left: 6px solid #4c8bf5;
}

.result-card {
    background: #ffffff;
    padding: 1.3rem 1.7rem;
    border-radius: 15px;
    border: 2px solid #4c8bf5;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
}

label {font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def cat_options(col):
    """Return sorted selectbox options with default first."""
    if train_df is not None and col in train_df.columns:
        vals = list(train_df[col].dropna().unique())
        default_val = defaults.get(col, "Unknown")

        if default_val not in vals:
            vals = [default_val] + vals

        vals = [v if pd.notna(v) else "Unknown" for v in vals]
        return sorted(set(vals), key=lambda x: (x != default_val, str(x).lower()))

    return [defaults.get(col, "Unknown")]


# Friendly UI labels
friendly_labels = {
    "opinion_seas_risk": "How risky do you believe the seasonal flu is?",
    "opinion_seas_vacc_effective": "How effective do you believe the flu vaccine is?",
    "hhs_geo_region": "Which HHS region do you live in?",
    "doctor_recc_seasonal": "Did a doctor recommend the flu vaccine?",
    "employment_occupation": "What is your job occupation?",
    "employment_industry": "What industry do you work in?",
    "education": "Your highest education level?",
}


# ---------------------------------------------------------
# TITLE HEADER
# ---------------------------------------------------------
st.markdown("""
<div style='text-align:center; padding-bottom:5px;'>
    <h1 style='margin-bottom:0;'>💉 Seasonal Flu Vaccine Predictor</h1>
    <p style='font-size:18px; color:#444;'>Answer a few questions and we’ll estimate your likelihood of getting the flu vaccine.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# AGE GROUP (Card Section)
# ---------------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("👤 Age Group")

col1, col2 = st.columns(2)

with col1:
    ag18 = st.radio("18–34 years", ["No", "Yes"], horizontal=True)
    ag35 = st.radio("35–44 years", ["No", "Yes"], horizontal=True)
    ag45 = st.radio("45–54 years", ["No", "Yes"], horizontal=True)

with col2:
    ag55 = st.radio("55–64 years", ["No", "Yes"], horizontal=True)
    ag65 = st.radio("65+ years", ["No", "Yes"], horizontal=True)

age_groups = {
    "18 - 34 Years": ag18,
    "35 - 44 Years": ag35,
    "45 - 54 Years": ag45,
    "55 - 64 Years": ag55,
    "65+ Years": ag65,
}

selected_groups = [age for age, ans in age_groups.items() if ans == "Yes"]

if len(selected_groups) == 0:
    age_group_value = defaults["age_group"]
elif len(selected_groups) == 1:
    age_group_value = selected_groups[0]
else:
    st.error("Please select **YES** for only one age group.")
    st.stop()

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# TOP-8 FEATURES (Card Section)
# ---------------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📋 Important Questions")

user_inputs = {}

# Two-column layout for cleaner UI
left, right = st.columns(2)

for i, feature in enumerate(top8):
    if feature == "age_group":
        user_inputs["age_group"] = age_group_value
        continue

    default_val = defaults.get(feature)
    label = friendly_labels.get(feature, feature)

    is_categorical = (
        isinstance(default_val, str) or
        (train_df is not None and feature in train_df and train_df[feature].dtype == object)
    )

    target_col = left if i % 2 == 0 else right

    with target_col:
        if is_categorical:
            options = cat_options(feature)
            user_inputs[feature] = st.selectbox(label, options)
        else:
            if default_val is None: default_val = 0.0
            user_inputs[feature] = st.number_input(label, value=float(default_val))

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# BUILD FULL INPUT VECTOR
# ---------------------------------------------------------
X_dict = {}
for feature in full_feature_list:
    if feature == "age_group":
        X_dict[feature] = age_group_value
    elif feature in user_inputs:
        X_dict[feature] = user_inputs[feature]
    else:
        X_dict[feature] = defaults.get(feature, 0)

X_df = pd.DataFrame([X_dict], columns=full_feature_list)


# ---------------------------------------------------------
# PREDICTION BUTTON + RESULTS CARD
# ---------------------------------------------------------
if st.button("🔍 Predict Seasonal Flu Vaccine Uptake", use_container_width=True):
    try:
        pred = model.predict(X_df)[0]
        proba = model.predict_proba(X_df)[0][1]

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.success(f"✔ You are **likely** to take the Seasonal Flu Vaccine.\n\n**Confidence: {proba:.2f}**")
        else:
            st.error(f"✘ You are **unlikely** to take the Seasonal Flu Vaccine.\n\n**Confidence: {proba:.2f}**")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed due to a preprocessing mismatch.")
        st.write("Error:", e)


# ---------------------------------------------------------
# DEBUG PANEL 
# ---------------------------------------------------------
with st.expander("🔧 Show Model Input Vector"):
    st.json(X_dict)

st.write("---")
st.caption(f"Top 8 features used: {top8}")
