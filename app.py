import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="EcoRate AI | Rohini Real Estate", 
    page_icon="🏠", 
    layout="wide" # Uses the full screen width
)

# Custom Styling
# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    </style>
    """, unsafe_allow_html=True) # Changed from unsafe_allow_value to unsafe_allow_html

# 2. Load the trained Pipeline
@st.cache_resource
def load_model():
    return joblib.load('ecorate_model.pkl')

pipeline = load_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🏠 Property Features")
with st.sidebar:
    loc = st.selectbox("Select Location", ["Rohini Sector 7", "Rohini Sector 13", "Pitampura", "Prashant Vihar"])
    area = st.number_input("Area (sqft)", value=1200, step=50)
    beds = st.slider("Bedrooms", 1, 5, 2)
    age = st.number_input("Property Age (Years)", value=5, min_value=0)
    metro = st.slider("Distance to Metro (km)", 0.0, 5.0, 0.5)
    solar = st.checkbox("Has Solar Panels?")
    
    st.divider()
    predict_btn = st.button("Calculate Market Value", use_container_width=True)

# --- MAIN PAGE ---
st.title("🏠 EcoRate: North Delhi AI Valuation")
st.info("Predicting property prices in Rohini using XGBoost & Sustainability Metrics.")

# Create Tabs for a cleaner UI
tab1, tab2, tab3 = st.tabs(["🚀 AI Predictor", "📊 Market Analytics", "🛠️ Engineering Roadmap"])
with tab1:
    if predict_btn:
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
        
        # Display Result in a big metric
        col_metric, _ = st.columns([1, 2])
        with col_metric:
            st.metric(label="Estimated Market Value", value=f"₹{price/100000:,.2f} Lakhs")
        
        st.divider()
        
        # SHAP Explanation
        st.subheader("🔍 Explainable AI: Why this price?")
        col_text, col_plot = st.columns([1, 2])
        
        with col_text:
            st.write("""
                This chart visualizes the **SHAP values**, showing how each feature contributed 
                to the final valuation. Positive values (green) increase the price, while 
                negative values (blue) decrease it.
            """)
        
        with col_plot:
            model = pipeline.named_steps['regressor']
            preprocessor = pipeline.named_steps['preprocessor']
            input_transformed = preprocessor.transform(input_df)
            
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_transformed)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.bar_plot(shap_vals[0], max_display=10, feature_names=preprocessor.get_feature_names_out(), show=False)
            st.pyplot(fig)
    else:
        st.write("👈 Fill in the property details in the sidebar and click **Calculate** to begin.")

with tab2:
    st.subheader("📈 Rohini Market Trends (Power BI)")
    st.write("""
        Explore the broader real estate market in Rohini through this Power BI dashboard. 
        It visualizes scraped data points across different sectors, price-per-sqft trends, 
        and eco-adoption rates.
    """)
    
    # PLACEHOLDER FOR YOUR IMAGE
    # Replace 'dashboard_screenshot.png' with your actual file name
    try:
        st.image("dashboard_screenshot.jpeg", caption="EcoRate Real Estate Market Intelligence Dashboard", use_container_width=True)
    except:
        st.warning("⚠️ Power BI Dashboard image not found. Please add 'dashboard_screenshot.png' to your project folder.")
with tab3:
    st.subheader("🏗️ How I Built EcoRate: The Technical Journey")
    st.write("A deep dive into the end-to-end pipeline of this project.")

    # Using st.expander for a "Step-by-Step" UI
    with st.expander("Step 1: Data Acquisition & Automation"):
        st.markdown("""
        * **Tooling:** BeautifulSoup4, Requests.
        * **Action:** Engineered a custom scraper to bypass pagination and extract real-time property listings from **MagicBricks**.
        * **Outcome:** Generated a localized dataset focused specifically on **Rohini**, Delhi.
        """)

    with st.expander("Step 2: Preprocessing & Feature Engineering"):
        st.markdown("""
        * **Tooling:** Scikit-Learn (Pipeline, ColumnTransformer).
        * **Action:** Automated the cleaning of raw scraped data, handling missing values via median imputation and encoding categorical variables.
        * **Highlight:** Created distance-based features (Metro proximity) to increase model accuracy.
        """)

    with st.expander("Step 3: Machine Learning Model (XGBoost)"):
        st.markdown("""
        * **Tooling:** XGBoost, Scikit-Learn.
        * **Action:** Trained an **XGBRegressor** to handle complex tabular data, achieving a high R² score.
        * **Persistence:** Serialized the final pipeline using **Joblib** for seamless deployment.
        """)

    with st.expander("Step 4: Explainable AI (SHAP)"):
        st.markdown("""
        * **Tooling:** SHAP (SHapley Additive exPlanations).
        * **Action:** Integrated Game Theory-based explanations to provide transparency, showing exactly how each feature (e.g., Solar Panels) moves the price.
        """)

    with st.expander("Step 5: Business Intelligence (Power BI)"):
        st.markdown("""
        * **Tooling:** Power BI Desktop, DAX.
        * **Action:** Built a geospatial dashboard to visualize market trends, price-per-sqft hubs, and eco-tech adoption rates in North Delhi.
        """)

    st.success("✅ **Deployment:** The project is fully hosted on **GitHub** and deployed via **Streamlit Cloud**.")        
