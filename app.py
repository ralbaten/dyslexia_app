import streamlit as st
import joblib
import json
import pandas as pd
from datetime import datetime

# --------------- Custom CSS for layout / styling ---------------

st.markdown(
    """
    <style>
        /* Global page styling */
        .main {
            padding-top: 1.5rem;
        }

        /* Hero title alignment */
        .app-header {
            padding: 0.5rem 0 1.25rem 0;
            border-bottom: 1px solid #ddd0b8;
            margin-bottom: 1rem;
        }

        /* Card style containers */
        .card {
            background-color: #fdf8f0;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #e2d6c3;
        }

        .card h3 {
            margin-top: 0;
        }

        /* Sidebar logo / header */
        [data-testid="stSidebar"] {
            background-color: #0a2342 !important;
        }

        [data-testid="stSidebar"] * {
            color: #f5f0e6 !important;
        }

        .sidebar-logo {
            font-size: 1.2rem;
            font-weight: 600;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #f5f0e655;
            margin-bottom: 0.75rem;
        }

        .sidebar-section-title {
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-top: 0.75rem;
            margin-bottom: 0.25rem;
            letter-spacing: 0.04em;
            color: #f9f3e0cc;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            background-color: #0a2342;
            color: #f5f0e6;
            border: none;
        }

        .stButton>button:hover {
            background-color: #132d55;
        }

        /* Make headers navy */
        h1, h2, h3, h4, h5, h6 {
            color: #0a2342 !important;
        }

        /* Footer text */
        .footer-text {
            font-size: 0.75rem;
            color: #777;
            text-align: center;
            margin-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------- Sidebar content ---------------

with st.sidebar:
    st.markdown('<div class="sidebar-logo">📚 Dyslexia AI Screener</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Overview</div>', unsafe_allow_html=True)
    st.write(
        "Prototype screening tool built for a graduate course project. "
        "It uses data from a computerized task and outputs a dyslexia risk score."
    )

    st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
    st.write("• Algorithm: XGBoost classifier")
    st.write("• Target: Dyslexia (Yes / No)")
    st.write("• Primary metric: AUC")

    st.markdown('<div class="sidebar-section-title">How to use</div>', unsafe_allow_html=True)
    st.write(
        "1. Enter the student's age.\n"
        "2. Keep typical task scores on (recommended), or adjust them if you have detailed data.\n"
        "3. Click **Predict** to see the risk score and interpretation."
    )

    st.markdown('<div class="sidebar-section-title">Project link</div>', unsafe_allow_html=True)
    st.markdown("[GitHub repo](https://github.com/ralbaten/dyslexia_app)")

# --------------- Load model and metadata ---------------

model = joblib.load("xgb_best_model.joblib")

with open("features.json") as f:
    features = json.load(f)

with open("feature_defaults.json") as f:
    default_values = json.load(f)

# --------------- Main header ---------------

st.markdown(
    """
    <div class="app-header">
        <h1>Dyslexia Screening App</h1>
        <p>
            This tool uses a machine learning model (XGBoost) trained on behavioral task data
            to estimate a student's likelihood of dyslexia. It is designed as a rapid,
            early-screening tool, not a diagnosis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------- Input card ---------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Input Student Data")
st.caption("Enter basic information. You can optionally adjust detailed task scores.")

inputs = {}

use_defaults = st.checkbox("Use typical task scores (recommended)", value=True)

age_default = float(default_values.get("Age", 10.0)) if use_defaults else 10.0
inputs["Age"] = st.number_input(
    "Age",
    value=age_default,
    step=1.0,
    min_value=5.0,
    max_value=18.0,
)

with st.expander("Advanced Task-Level Inputs (Optional)"):
    st.caption("Defaults are typical values from the training data.")
    for feature in features:
        if feature == "Age":
            continue
        base_val = float(default_values.get(feature, 0.0)) if use_defaults else 0.0
        inputs[feature] = st.number_input(feature, value=base_val)

predict_clicked = st.button("Predict")
st.markdown('</div>', unsafe_allow_html=True)  # close input card

# --------------- Results card ---------------

if predict_clicked:
    input_df = pd.DataFrame([inputs])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Results")

    st.write("Predicted Dyslexia Class (1 = Yes, 0 = No):", int(pred))
    st.write(f"Model probability of dyslexia: **{prob:.3f}**")

    if prob < 0.3:
        risk_level = "Low"
        st.success("Risk level: Low. Pattern looks similar to non-dyslexic students in the dataset.")
    elif prob < 0.6:
        risk_level = "Moderate"
        st.warning("Risk level: Moderate. This pattern appears in both dyslexic and non-dyslexic students.")
    else:
        risk_level = "High"
        st.error("Risk level: High. This pattern is similar to students labeled with dyslexia in the dataset.")

    st.write("Risk score visualization:")
    st.progress(float(prob))

    with st.expander("Which features matter most overall?"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"feature": features, "importance": importances})
        fi_top = fi_df.sort_values("importance", ascending=False).head(10)
        st.write("Top 10 features the model relies on the most:")
        st.bar_chart(fi_top.set_index("feature"))

    st.markdown("### Export this result")
    result_df = pd.DataFrame(
        [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "age": inputs["Age"],
                "predicted_class": int(pred),
                "probability_dyslexia": float(prob),
                "risk_level": risk_level,
            }
        ]
    )
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download result as CSV",
        data=csv_bytes,
        file_name="dyslexia_screening_result.csv",
        mime="text/csv",
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close results card

# --------------- Info / interpretation card ---------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("How to interpret these results")
st.write(
    "- The model output is a risk estimate based on patterns in the training data, "
    "not a medical or educational diagnosis.\n"
    "- Low risk means the pattern looks similar to students without a dyslexia label in the dataset.\n"
    "- Moderate risk means the pattern overlaps both groups and may warrant closer monitoring.\n"
    "- High risk suggests the pattern is similar to students who were labeled with dyslexia in the dataset.\n"
    "- Any concerns should be followed up with formal assessments by qualified professionals such as "
    "school psychologists, special educators, or clinicians."
)
st.markdown('</div>', unsafe_allow_html=True)

# --------------- Footer ---------------

st.markdown(
    '<div class="footer-text">Research prototype for educational purposes only.</div>',
    unsafe_allow_html=True,
)
