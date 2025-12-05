# Dyslexia Screening App

This repository contains a Streamlit web app and machine learning pipeline for early dyslexia risk screening.  
The model is trained on a behavioral task dataset where students complete multiple trials (clicks, hits, misses, accuracy, etc.), and outputs a dyslexia risk probability.

> **Note:** This is a research prototype built for a graduate course project.  
> It is not a clinical tool and should not be used as a diagnosis.

---

## 1. Business problem

Early identification of dyslexia is critical for giving students timely support, but many schools do not have access to frequent, low-cost screening tools.

This project explores whether a lightweight, browser-based tool can:
- Estimate dyslexia risk from task performance data
- Help teachers flag students for further assessment
- Provide an interpretable risk score instead of a black-box label

---

## 2. Data & features

- Behavioral task dataset with ~3,600 students
- Features: age + 30+ task trials (clicks, hits, misses, accuracy, score, etc.)
- Target: `Dyslexia` (Yes / No), converted to 0/1

Preprocessing steps:
- Dropped non-task demographic fields (e.g., gender, language)
- Train/test split with stratification due to class imbalance
- Checked class imbalance and evaluation beyond raw accuracy

---

## 3. Modeling

Models explored:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (final model)

The final model is an **XGBoost classifier** tuned with `RandomizedSearchCV` (AUC as the metric).  
Key evaluation metrics:
- Test AUC ≈ _[fill in your final AUC]_  
- Precision/recall for both classes
- Focus on recall for the dyslexia (positive) class

Feature importance is computed from the fitted XGBoost model to understand which task trials matter most.

---

## 4. App overview

The Streamlit app allows:

1. Entering a student's age
2. Optionally overriding task-level performance scores  
   (defaults are set to typical values from the training data)
3. Generating:
   - A predicted class (1 = dyslexia risk, 0 = low risk)
   - A probability of dyslexia
   - A qualitative risk level (low / moderate / high)
   - A downloadable CSV and PDF report
4. Viewing:
   - Risk score visualization
   - Top features the model relies on globally


