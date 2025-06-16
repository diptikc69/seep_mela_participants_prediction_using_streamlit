import streamlit as st
import numpy as np
import joblib  # Use joblib to load sklearn models

st.set_page_config(page_title="SEEP Mela Participant Predictor", layout="centered")
st.title("🎯 SEEP Mela Participant Predictor")

# Model selection dropdown
model_choice = st.selectbox("Select Model", ["Baseline Model", "Fine-tuned Model"])
model_path = "base_model.pkl" if model_choice == "Baseline Model" else "final_tuned_model.pkl"

# Load model
try:
    model = joblib.load(model_path)
    # st.write(f"✅ Loaded model: **{model_path}**")

except FileNotFoundError:
    st.error(f"❌ Model file '{model_path}' not found.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# User inputs
st.subheader("📋 Input Features")
course_encoded = st.number_input("Course (encoded)", min_value=0, max_value=5, step=1)
sessions = st.number_input("Sessions Conducted", min_value=1, max_value=10, step=1)
year = st.number_input("Academic Year (e.g., 2083)", min_value=2083, max_value=2090, step=1)

# Predict button
if st.button("🎯 Predict Participants"):
    features = np.array([[course_encoded, sessions, year]])
    try:
        prediction = model.predict(features)
        st.success(f"📊 Predicted Participants: **{int(prediction[0])}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
