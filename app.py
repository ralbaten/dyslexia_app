import streamlit as st
import joblib
import json
import pandas as pd

# ---- Page title & description ----
st.title("Dyslexia Screening App")

st.markdown(
    "This tool uses a machine learning model (XGBoost) trained on behavioral task data "
    "to estimate a student's likelihood of dyslexia. "
    "It is designed as a rapid, early-screening tool, not a diagnosis."
)

st.markdown("---")

# ---- Sidebar: About the app ----
with st.sidebar:
    st.header("About this app")
    st.write(
        "This app is a prototype screening tool built for a course project. "
        "It uses data from a computerized task where students complete multiple trials "
        "and generates a dyslexia risk score."
    )

    st.subheader("Model details")
    st.write("• Algorithm: XGBoost classifier")
    st.write("• Target: Dyslexia (Yes / No)")
    st.write("• Metric used during training: AUC")

    st.subheader("How to use")
    st.write(
        "1. Enter the student's age.\n"
        "2. Leave typical task scores on (recommended), or adjust them if you have detailed data.\n"
        "3. Click Predict to see the risk score."
    )

    st.subheader("Project links")
    st.write("GitHub repo (example):")
    st.markdown("[ralbaten/dyslexia_app](https://github.com/ralbaten/dyslexia_app)")

# ---- Load model and metadata ----
model = joblib.load("xgb_best_model.joblib")

# Load feature names
with open("features.json") as f:
    features = json.load(f)

# Load typical/default values from training data
with open("feature_defaults.json") as f:
    default_values = json.load(f)

# ---- Input section ----
inputs = {}

st.subheader("Input Student Data")
st.caption("Enter age. You can optionally adjust detailed task scores.")

# Checkbox to use typical defaults
use_defaults = st.checkbox("Use typical task scores (recommended)", value=True)

# Age at the top, using default mean if available
age_default = float(default_values.get("Age", 10.0)) if use_defaults else 10.0
inputs["Age"] = st.number_input("Age", value=age_default, step=1.0)

# Advanced features in an expander
with st.expander("Advanced Task-Level Inputs (Optional)"):
    for feature in features:
        if feature == "Age":
            continue

        # Use default mean if available and checkbox is on, otherwise 0
        base_val = float(default_values.get(feature, 0.0)) if use_defaults else 0.0
        inputs[feature] = st.number_input(feature, value=base_val)

# ---- Prediction button and results ----
if st.button("Predict"):
    # Build dataframe from inputs
    input_df = pd.DataFrame([inputs])

    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Results section
    st.markdown("---")
    st.subheader("Results")

    st.write("Predicted Dyslexia Class (1 = Yes, 0 = No):", int(pred))
    st.write(f"Model probability of dyslexia: {prob:.3f}")

    # Risk band
    if prob < 0.3:
        risk_level = "Low"
        st.success(
            "Risk level: Low. Pattern looks similar to non-dyslexic students in the dataset."
        )
    elif prob < 0.6:
        risk_level = "Moderate"
        st.warning(
            "Risk level: Moderate. This pattern appears in both dyslexic and non-dyslexic students."
        )
    else:
        risk_level = "High"
        st.error(
            "Risk level: High. This pattern is similar to students labeled with dyslexia in the dataset."
        )

    # Visual bar for probability
    st.write("Risk score visualization:")
    st.progress(float(prob))

# ---- Interpretation / disclaimer section ----
with st.expander("How to interpret these results"):
    st.write(
        "- The model output is a risk estimate based on patterns in the training data, "
        "not a medical or educational diagnosis.\n"
        "- Low risk means the pattern looks similar to students without a dyslexia label in the dataset.\n"
        "- Moderate risk means the pattern overlaps both groups and may warrant closer monitoring.\n"
        "- High risk suggests the pattern is similar to students who were labeled with dyslexia in the dataset.\n"
        "- Any concerns should be followed up with formal assessments by qualified professionals."
    )
