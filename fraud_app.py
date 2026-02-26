import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ==============================
# Load Model Package
# ==============================

import joblib

tfidf = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("onehot_encoder.pkl")
svm_model = joblib.load("linear_svm_model.pkl")

# ==============================
# Define Columns
# ==============================

cat_cols = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

meta_cols = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "salary_provided",
    "location_missing",
    "company_profile_missing",
    "has_benefits"
]

# ==============================
# Text Cleaning Function
# (Use SAME cleaning as training)
# ==============================

import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

# ==============================
# Streamlit UI
# ==============================

st.title("🚨 Job Fraud Detection System")
st.write("Enter job details below to check if it is Fraud or Real.")

# Text Inputs
title = st.text_input("Job Title")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
company_profile = st.text_area("Company Profile")
benefits = st.text_area("Benefits")
location = st.text_input("Location")

# Dropdown Inputs
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other"])
required_experience = st.selectbox("Experience Level", ["Entry level", "Mid level", "Senior level", "Director", "Executive"])
required_education = st.selectbox("Education Level", ["High School", "Bachelor's Degree", "Master's Degree", "PhD", "Other"])
industry = st.text_input("Industry")
function = st.text_input("Function")

# Binary Meta Inputs
telecommuting = st.selectbox("Telecommuting", [0, 1])
has_company_logo = st.selectbox("Company Logo Provided", [0, 1])
has_questions = st.selectbox("Has Screening Questions", [0, 1])
salary_provided = st.selectbox("Salary Provided", [0, 1])

# ==============================
# Prediction Button
# ==============================

if st.button("Check Fraud"):

    new_job = {
        "title": title,
        "description": description,
        "requirements": requirements,
        "company_profile": company_profile,
        "benefits": benefits,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "location": location
    }

    new_df = pd.DataFrame([new_job])

    # Combine text
    text_columns = ["title", "description", "requirements"]
    new_df["combined_text_raw"] = new_df[text_columns].fillna("").agg(" ".join, axis=1)
    new_df["clean_text"] = new_df["combined_text_raw"].apply(clean_text)

    # TFIDF
    X_text = tfidf.transform(new_df["clean_text"])

    # Handle categorical
    for col in cat_cols:
        if col not in new_df.columns:
            new_df[col] = "Unknown"
        else:
            new_df[col] = new_df[col].fillna("Unknown")

    X_cat = encoder.transform(new_df[cat_cols])

    # Meta Features
    new_df["telecommuting"] = telecommuting
    new_df["has_company_logo"] = has_company_logo
    new_df["has_questions"] = has_questions
    new_df["salary_provided"] = salary_provided
    new_df["location_missing"] = 0 if location else 1
    new_df["company_profile_missing"] = 0 if company_profile else 1
    new_df["has_benefits"] = 0 if benefits else 1

    X_meta = csr_matrix(new_df[meta_cols].values)

    # Combine
    X_final = hstack([X_text, X_cat, X_meta])

    # Predict
    probability = svm_model.predict_proba(X_final)[:, 1][0]
    prediction = 1 if probability >= threshold else 0

    st.subheader("Result")

    st.write("Fraud Probability:", round(probability, 4))

    if prediction == 1:
        st.error("⚠️ This Job Posting is Likely FRAUD")
    else:
        st.success("✅ This Job Posting Appears Legitimate")

