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

st.subheader("Input Student Data")

inputs = {}

# Keep Age visible at the top
inputs["Age"] = st.number_input("Age", value=10.0, step=1.0)

# Group the rest in a collapsible section
with st.expander("Advanced task-level inputs"):
    for feature in features:
        if feature == "Age":
            continue
        inputs[feature] = st.number_input(feature, value=0.0)


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

