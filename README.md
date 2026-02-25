# 🏠 EcoRate: AI-Powered Real Estate Sustainability & Price Predictor

**EcoRate** is an end-to-end Machine Learning application designed to predict residential property prices in **Rohini, Delhi**. Unlike standard price predictors, this project integrates sustainability metrics—such as solar potential and metro proximity—and uses **Explainable AI (SHAP)** to justify every prediction.

🚀 **[Live Demo Link](https://ecorate-predictor.streamlit.app/)**
<img width="1103" height="621" alt="Screenshot 2026-02-24 204214" src="https://github.com/user-attachments/assets/29978f67-3c7f-4a83-95f6-03a926548f9c" />



---



## 🌟 Key Features
* **Custom Data Acquisition:** Utilizes a custom-built Python scraper (BeautifulSoup) to gather real-time listings from major real estate portals.
* **Production-Grade Pipeline:** Implements a Scikit-Learn `Pipeline` with `ColumnTransformer` for robust preprocessing, ensuring no data leakage.
* **Advanced Modeling:** Uses **XGBoost Regression**, optimized for high-performance tabular data prediction.
* **Model Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** values to show users exactly how features like "Distance to Metro" or "Solar Panels" impact their property value.
* **Interactive Dashboard:** A clean, user-friendly interface built with **Streamlit** for real-time price estimation.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Data Science:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Explainability:** SHAP
* **Web Scraping:** BeautifulSoup4, Requests
* **Deployment:** Streamlit Cloud, GitHub
* **Environment Management:** Python venv

---

## 📊 Model Performance
The model was evaluated using standard regression metrics to ensure reliability for end-users:
* **R² Score:** 0.XX (Insert your actual R2 score here)
* **Mean Absolute Error (MAE):** ₹X.XX Lakhs (Insert your actual MAE here)

---

## 📂 Project Structure
```text
├── app.py              # Streamlit Web Application
├── train_model.py      # ML Pipeline & Model Training script
├── scraper.py          # Web Scraper for data collection
├── ecorate_model.pkl   # Trained Pipeline object (Serialized)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
