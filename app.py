import streamlit as st
import joblib
import json
import pandas as pd

st.title("Dyslexia Screening App")
st.write("This app uses an XGBoost model trained on behavioral task data to estimate dyslexia risk.")

# Load model
model = joblib.load("xgb_best_model.joblib")

# Load feature names
with open("features.json") as f:
    features = json.load(f)

# Load typical/default values from training data
with open("feature_defaults.json") as f:
    default_values = json.load(f)


st.subheader("Input Student Data")

inputs = {}

st.subheader("Input Student Data")

# Checkbox to use typical defaults
use_defaults = st.checkbox("Use typical task scores (recommended)", value=True)

# Age at the top, using default mean if available
age_default = float(default_values.get("Age", 10.0)) if use_defaults else 10.0
inputs["Age"] = st.number_input("Age", value=age_default, step=1.0)

# Advanced features in an expander
with st.expander("Advanced task-level inputs"):
    for feature in features:
        if feature == "Age":
            continue

        # Use default mean if available and checkbox is on, otherwise 0
        base_val = float(default_values.get(feature, 0.0)) if use_defaults else 0.0
        inputs[feature] = st.number_input(feature, value=base_val)


input_df = pd.DataFrame([inputs])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Results")
    st.write("Predicted Dyslexia Class (1 = Yes, 0 = No):", int(pred))
    st.write(f"Probability of Dyslexia: {prob:.3f}")

    if pred == 1:
        st.warning("Possible dyslexia risk detected.")
    else:
        st.success("Low dyslexia risk.")


