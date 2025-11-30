import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("Breast_Cancer_ML_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the tumor features below (raw values).")

# Feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

col1, col2 = st.columns(2)

inputs = []
for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        value = col1.number_input(f"{feature}", value=0.0)
    else:
        value = col2.number_input(f"{feature}", value=0.0)
    inputs.append(value)

if st.button("🔍 Predict"):
    # Convert to array
    input_array = np.array(inputs).reshape(1, -1)
    
    # Scale using loaded scaler
    scaled_input = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(scaled_input)[0]

    if prediction == 0:
        st.error("🔴 **Prediction: Malignant (M)**")
    else:
        st.success("🟢 **Prediction: Benign (B)**")
st.write("Developed by Ritik Kumar")