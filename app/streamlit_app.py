import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown("Detect whether a transaction is **Fraudulent or Normal** using Machine Learning.")

# Sidebar
st.sidebar.header("Input Transaction Features")

def user_input():
    features = []

    for i in range(30):
        val = st.sidebar.number_input(f"Feature V{i+1}", value=0.0)
        features.append(val)

    return np.array(features).reshape(1, -1)

input_data = user_input()

# Predict button
if st.button("🔍 Predict Transaction"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")

# -------------------------
# Dataset Visualization
# -------------------------

st.subheader("📊 Dataset Overview")

df = pd.read_csv("data/creditcard.csv")

st.write(df.head())

# Fraud Distribution
st.subheader("📉 Fraud vs Normal Distribution")

fraud_count = df['Class'].value_counts()

st.bar_chart(fraud_count)

# -------------------------
# Insights Section
# -------------------------

st.subheader("📌 Insights")

st.markdown("""
- Dataset is highly imbalanced  
- Fraud transactions are very rare  
- Model focuses on **Recall** to catch fraud cases  
""")
if st.button("🎲 Generate Random Transaction"):
    random_data = np.random.rand(1, 30)
    prediction = model.predict(random_data)

    st.write("Generated Transaction:", random_data)

    if prediction[0] == 1:
        st.error("⚠ Fraud Detected")
    else:
        st.success("✅ Normal")