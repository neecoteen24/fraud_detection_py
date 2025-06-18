import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("fraud_detection_pipeline.pkl")

# App title and description
st.set_page_config(page_title="Fraud Detection App", page_icon="💳")
st.title("💰 Fraud Detection Prediction App")
st.markdown("Fill in the transaction details below to check if it is **fraudulent** or **legitimate**.")

st.divider()

# Input Fields in two columns
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"],
        help="Select the type of financial transaction."
    )
    amount = st.number_input("💵 Amount", min_value=0.0, value=1000.0, help="Enter the amount of transaction.")
    oldbalanceOrg = st.number_input("🏦 Old Balance (Sender)", min_value=0.0, value=10000.0, help="Balance before transaction.")

with col2:
    newbalanceOrig = st.number_input("🏦 New Balance (Sender)", min_value=0.0, value=9000.0, help="Balance after transaction.")
    oldbalanceDest = st.number_input("🏦 Old Balance (Receiver)", min_value=0.0, value=0.0, help="Receiver's balance before transaction.")
    newbalanceDest = st.number_input("🏦 New Balance (Receiver)", min_value=0.0, value=0.0, help="Receiver's balance after transaction.")

st.markdown("---")

# Prediction button
if st.button("🔍 Predict Transaction"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    try:
        # Make prediction and get probability
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # Probability of being fraud

        st.subheader("🔎 Prediction Result")
        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction Detected!\n\n**Confidence: {proba:.2%}**")
        else:
            st.success(f"✅ Legitimate Transaction\n\n**Confidence: {1 - proba:.2%}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
