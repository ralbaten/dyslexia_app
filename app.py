import streamlit as st
import joblib
import json
import pandas as pd
from datetime import datetime

# ---------------- Page title & description ----------------

st.title("Dyslexia Screening App")

st.markdown(
    "This tool uses a machine learning model (XGBoost) trained on behavioral task data "
    "to estimate a student's likelihood of dyslexia. "
    "It is designed as a rapid, early-screening tool, not a diagnosis."
)

st.markdown("---")

# ---------------- Sidebar: About the app ----------------

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
    st.write("• Primary evaluation metric: AUC")

    st.subheader("How to use")
    st.write(
        "1. Enter the student's age.\n"
        "2. Leave typical task scores on (recommended), or adjust them if you have detailed data.\n"
        "3. Click **Predict** to see the risk score and interpretation."
    )

    st.subheader("Project links")
    st.markdown("[GitHub repo](https://github.com/ralbaten/dyslexia_app)")

# ---------------- Load model and metadata ----------------

model = joblib.load("xgb_best_model.joblib")

# Load feature names
with open("features.json") as f:
    features = json.load(f)

# Load typical/default values from training data
with open("feature_defaults.json") as f:
    default_values = json.load(f)

# ---------------- Input section ----------------

inputs = {}

st.subheader("Input Student Data")
st.caption("Enter age. You can optionally adjust detailed task scores.")

# Checkbox to use typical defaults
use_defaults = st.checkbox("Use typical task scores (recommended)", value=True)

# Age at the top, using default mean if available
age_default = float(default_values.get("Age", 10.0)) if use_defaults else 10.0
inputs["Age"] = st.number_input("Age", value=age_default, step=1.0, min_value=5.0, max_value=18.0)

# Advanced features in an expander
with st.expander("Advanced Task-Level Inputs (Optional)"):
    st.caption("These values come from the computerized task. Defaults are typical values from the training data.")
    for feature in features:
        if feature == "Age":
            continue

        # Use default mean if available and checkbox is on, otherwise 0
        base_val = float(default_values.get(feature, 0.0)) if use_defaults else 0.0
        inputs[feature] = st.number_input(feature, value=base_val)

# ---------------- Prediction button and results ----------------

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
    st.write(f"Model probability of dyslexia: **{prob:.3f}**")

    # Risk band + message
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

    # Overall model feature importance
    with st.expander("Which features matter most overall?"):
        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": features,
            "importance": importances
        })

        fi_top = fi_df.sort_values("importance", ascending=False).head(10)

        st.write("Top 10 features the model relies on the most:")
        st.bar_chart(fi_top.set_index("feature"))

    # Downloadable summary for this prediction
    st.markdown("### Export this result")
    result_df = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        "age": inputs["Age"],
        "predicted_class": int(pred),
        "probability_dyslexia": float(prob),
        "risk_level": risk_level
    }])

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download result as CSV",
        data=csv_bytes,
        file_name="dyslexia_screening_result.csv",
        mime="text/csv"
    )

# ---------------- Interpretation / disclaimer ----------------

with st.expander("How to interpret these results"):
    st.write(
        "- The model output is a **risk estimate** based on patterns in the training data, "
        "not a medical or educational diagnosis.\n"
        "- **Low risk** means the pattern looks similar to students without a dyslexia label in the dataset.\n"
        "- **Moderate risk** means the pattern overlaps both groups and may warrant closer monitoring.\n"
        "- **High risk** suggests the pattern is similar to students who were labeled with dyslexia in the dataset.\n"
        "- Any concerns should be followed up with formal assessments by qualified professionals "
        "such as school psychologists, special educators, or clinicians."
    )

st.markdown("---")
st.caption("Research prototype for educational purposes only.")

# Inject custom CSS for navy + beige theme
st.markdown("""
    <style>
        /* Main page background */
        .stApp {
            background-color: #f5f1e6;
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #0b1f3a !important; /* navy */
        }

        /* Sidebar text */
        section[data-testid="stSidebar"] * {
            color: #f5f1e6 !important; /* beige text */
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #0b1f3a !important; /* navy */
        }

        /* Links */
        a {
            color: #0b1f3a !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            color: #0b1f3a !important;
            font-weight: 600;
        }

        /* Buttons */
        div.stButton button {
            background-color: #0b1f3a !important;
            color: #f5f1e6 !important;
            border-radius: 6px;
            bord

