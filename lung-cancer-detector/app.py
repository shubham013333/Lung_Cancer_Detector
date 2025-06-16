
import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("models/lung_model.pkl")
columns = joblib.load("models/feature_columns.pkl")

st.title("🩺 Lung Cancer Prediction App")

with st.form("lung_form"):
    GENDER = st.selectbox("Gender", ["MALE", "FEMALE"])
    AGE = st.slider("Age", 20, 100, 45)
    SMOKING = st.selectbox("Smoking?", ["YES", "NO"])
    YELLOW_FINGERS = st.selectbox("Yellow Fingers?", ["YES", "NO"])
    ANXIETY = st.selectbox("Anxiety?", ["YES", "NO"])
    PEER_PRESSURE = st.selectbox("Peer Pressure?", ["YES", "NO"])
    CHRONIC_DISEASE = st.selectbox("Chronic Disease?", ["YES", "NO"])
    FATIGUE = st.selectbox("Fatigue?", ["YES", "NO"])
    ALLERGY = st.selectbox("Allergy?", ["YES", "NO"])
    WHEEZING = st.selectbox("Wheezing?", ["YES", "NO"])
    ALCOHOL_CONSUMING = st.selectbox("Alcohol Consuming?", ["YES", "NO"])
    COUGHING = st.selectbox("Coughing?", ["YES", "NO"])
    SHORTNESS_OF_BREATH = st.selectbox("Shortness of Breath?", ["YES", "NO"])
    SWALLOWING_DIFFICULTY = st.selectbox("Swallowing Difficulty?", ["YES", "NO"])
    CHEST_PAIN = st.selectbox("Chest Pain?", ["YES", "NO"])

    submit = st.form_submit_button("Predict")

if submit:
    # Prepare input
    input_data = {
        "GENDER": 1 if GENDER == "MALE" else 0,
        "AGE": AGE,
        "SMOKING": 1 if SMOKING == "YES" else 0,
        "YELLOW_FINGERS": 1 if YELLOW_FINGERS == "YES" else 0,
        "ANXIETY": 1 if ANXIETY == "YES" else 0,
        "PEER_PRESSURE": 1 if PEER_PRESSURE == "YES" else 0,
        "CHRONIC DISEASE": 1 if CHRONIC_DISEASE == "YES" else 0,
        "FATIGUE": 1 if FATIGUE == "YES" else 0,
        "ALLERGY": 1 if ALLERGY == "YES" else 0,
        "WHEEZING": 1 if WHEEZING == "YES" else 0,
        "ALCOHOL CONSUMING": 1 if ALCOHOL_CONSUMING == "YES" else 0,
        "COUGHING": 1 if COUGHING == "YES" else 0,
        "SHORTNESS OF BREATH": 1 if SHORTNESS_OF_BREATH == "YES" else 0,
        "SWALLOWING DIFFICULTY": 1 if SWALLOWING_DIFFICULTY == "YES" else 0,
        "CHEST PAIN": 1 if CHEST_PAIN == "YES" else 0,
    }

    # Convert to DataFrame and align columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=columns)

    prediction = model.predict(input_df)[0]
    st.write("🧪 Prediction Result:", "🟥 **Lung Cancer Detected**" if prediction == 1 else "🟩 **No Lung Cancer**")
