import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="EcoRate AI", page_icon="🏠")

# 2. Load the trained Pipeline
@st.cache_resource
def load_model():
    return joblib.load('ecorate_model.pkl')

pipeline = load_model()

st.title("🏠 EcoRate: Rohini Real Estate AI")
st.markdown("Predicting property prices in North Delhi using AI & Sustainability metrics.")

# 3. Input Form
with st.form("property_form"):
    st.subheader("Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (sqft)", value=1200, step=50)
        beds = st.slider("Bedrooms", 1, 5, 2)
        loc = st.selectbox("Location", ["Rohini Sector 7", "Rohini Sector 13", "Pitampura", "Prashant Vihar"])
    
    with col2:
        age = st.number_input("Property Age (Years)", value=5, min_value=0)
        metro = st.slider("Distance to Metro (km)", 0.0, 5.0, 0.5)
        solar = st.checkbox("Has Solar Panels installed?")

    submitted = st.form_submit_state = st.form_submit_button("Predict Market Price")

# 4. Prediction & Explanation
if submitted:
    # Create input DataFrame
    input_df = pd.DataFrame({
        'Area_sqft': [area],
        'Bedrooms': [beds],
        'Location': [loc],
        'Age_of_Property': [age],
        'Solar_Panel': [1 if solar else 0],
        'Distance_to_Metro_km': [metro]
    })
    
    # Get Prediction
    price = pipeline.predict(input_df)[0]
    
    st.divider()
    st.metric(label="Estimated Market Value", value=f"₹{price/100000:,.2f} Lakhs")
    
    # 5. Explain with SHAP
    st.subheader("🔍 Why this price?")
    st.write("This chart shows which factors increased or decreased the estimated value.")
    
    # Prepare data for SHAP
    model = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']
    input_transformed = preprocessor.transform(input_df)
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_transformed)
    
    # Plotting
    fig, ax = plt.subplots()
    shap.bar_plot(shap_vals[0], max_display=10, feature_names=preprocessor.get_feature_names_out())
    st.pyplot(fig)