import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# UI Elements
st.title("🚨 Fraud Detection AML System")
st.markdown("Advanced machine learning-based fraud detection for financial transactions")
st.divider()

# Sidebar for model information
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write("""
    **Algorithm:** Logistic Regression with balanced class weights
    
    **Features:** 10 engineered features including:
    - Transaction type (one-hot encoded)
    - Amount and balances
    - Balance differences
    - Amount-to-balance ratio
    - Account emptying indicator
    
    **Performance:** 95%+ accuracy with 94% fraud recall
    """)
    
    st.header("📊 Risk Indicators")
    st.write("""
    🔴 **High Risk:**
    - TRANSFER/CASH_OUT transactions
    - Large amounts (>$100k)
    - Account emptying
    - High amount-to-balance ratio
    
    🟡 **Medium Risk:**
    - Medium amounts ($50k-$100k)
    - Unusual balance patterns
    
    🟢 **Low Risk:**
    - PAYMENT/DEPOSIT transactions
    - Small amounts (<$50k)
    """)

# Main form
st.header("📝 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type", 
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"],
        help="TRANSFER and CASH_OUT have higher fraud risk"
    )
    
    amount = st.number_input(
        "Amount ($)", 
        min_value=0.0, 
        value=1000.0, 
        step=0.01,
        help="Transaction amount in dollars"
    )

with col2:
    oldbalanceOrg = st.number_input(
        "Sender Old Balance ($)", 
        min_value=0.0, 
        value=10000.0, 
        step=0.01
    )
    
    newbalanceOrig = st.number_input(
        "Sender New Balance ($)", 
        min_value=0.0, 
        value=9000.0, 
        step=0.01
    )

# Receiver information
st.subheader("Receiver Information")
col3, col4 = st.columns(2)

with col3:
    oldbalanceDest = st.number_input(
        "Receiver Old Balance ($)", 
        min_value=0.0, 
        value=0.0, 
        step=0.01
    )

with col4:
    newbalanceDest = st.number_input(
        "Receiver New Balance ($)", 
        min_value=0.0, 
        value=0.0, 
        step=0.01
    )

# Validation and risk assessment
st.divider()
st.header("🔍 Transaction Analysis")

# Calculate derived features
balanceDiffOrig = oldbalanceOrg - newbalanceOrig
balanceDiffDest = newbalanceDest - oldbalanceDest
amount_to_balance_ratio = amount / (oldbalanceOrg + 1)  # +1 to avoid division by zero
is_account_emptied = 1 if (oldbalanceOrg > 0 and newbalanceOrig == 0) else 0

# Validation checks
expected_newbalanceOrig = oldbalanceOrg - amount
expected_newbalanceDest = oldbalanceDest + amount

balance_org_valid = abs(newbalanceOrig - expected_newbalanceOrig) < 0.01
balance_dest_valid = abs(newbalanceDest - expected_newbalanceDest) < 0.01

# Display validation results
col5, col6 = st.columns(2)

with col5:
    st.subheader("✅ Validation Results")
    if balance_org_valid:
        st.success("Sender balance: ✓ Consistent")
    else:
        st.error(f"Sender balance: ✗ Inconsistent")
        st.info(f"Expected: ${expected_newbalanceOrig:.2f}, Entered: ${newbalanceOrig:.2f}")
    
    if balance_dest_valid:
        st.success("Receiver balance: ✓ Consistent")
    else:
        st.error(f"Receiver balance: ✗ Inconsistent")
        st.info(f"Expected: ${expected_newbalanceDest:.2f}, Entered: ${newbalanceDest:.2f}")

with col6:
    st.subheader("📊 Derived Features")
    st.metric("Balance Difference (Sender)", f"${balanceDiffOrig:.2f}")
    st.metric("Balance Difference (Receiver)", f"${balanceDiffDest:.2f}")
    st.metric("Amount/Balance Ratio", f"{amount_to_balance_ratio:.3f}")
    st.metric("Account Emptied", "Yes" if is_account_emptied else "No")

# Risk assessment
st.subheader("⚠️ Risk Assessment")
risk_score = 0
risk_factors = []

# Transaction type risk
if transaction_type in ["TRANSFER", "CASH_OUT"]:
    risk_score += 3
    risk_factors.append("🔴 High-risk transaction type (fraud typically occurs in TRANSFER/CASH_OUT)")
else:
    risk_factors.append("🟢 Low-risk transaction type")

# Amount risk
if amount > 100000:
    risk_score += 3
    risk_factors.append("🔴 Large transaction amount (>$100k)")
elif amount > 50000:
    risk_score += 2
    risk_factors.append("🟡 Medium transaction amount ($50k-$100k)")
else:
    risk_factors.append("🟢 Small transaction amount (<$50k)")

# Account emptying risk
if is_account_emptied and transaction_type in ["TRANSFER", "CASH_OUT"]:
    risk_score += 4
    risk_factors.append("🔴 Account emptied after transaction (high fraud indicator)")

# Amount to balance ratio risk
if amount_to_balance_ratio > 0.8:
    risk_score += 3
    risk_factors.append("🔴 High amount-to-balance ratio (>80%)")
elif amount_to_balance_ratio > 0.5:
    risk_score += 2
    risk_factors.append("🟡 Medium amount-to-balance ratio (50-80%)")

# Balance validation risk
if not balance_org_valid or not balance_dest_valid:
    risk_score += 2
    risk_factors.append("🟡 Balance inconsistencies detected")

# Display risk factors
for factor in risk_factors:
    st.write(factor)

# Overall risk level
st.subheader("🎯 Overall Risk Level")
if risk_score >= 7:
    risk_level = "🔴 HIGH RISK"
    risk_color = "red"
elif risk_score >= 4:
    risk_level = "🟡 MEDIUM RISK"
    risk_color = "orange"
else:
    risk_level = "🟢 LOW RISK"
    risk_color = "green"

st.markdown(f"<h3 style='color: {risk_color};'>{risk_level}</h3>", unsafe_allow_html=True)
st.write(f"Risk Score: {risk_score}/10")

# Prediction section
st.divider()
st.header("🤖 AI Prediction")

if st.button("🔮 Predict Fraud", type="primary", use_container_width=True):
    # Use corrected balances if validation failed
    if not balance_org_valid:
        newbalanceOrig = expected_newbalanceOrig
        st.info(f"Using corrected sender balance: ${newbalanceOrig:.2f}")
    
    if not balance_dest_valid:
        newbalanceDest = expected_newbalanceDest
        st.info(f"Using corrected receiver balance: ${newbalanceDest:.2f}")
    
    # Recalculate derived features with corrected values
    balanceDiffOrig = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest
    amount_to_balance_ratio = amount / (oldbalanceOrg + 1)
    is_account_emptied = 1 if (oldbalanceOrg > 0 and newbalanceOrig == 0) else 0
    
    # Prepare input data with all features
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOrig": balanceDiffOrig,
        "balanceDiffDest": balanceDiffDest,
        "amount_to_balance_ratio": amount_to_balance_ratio,
        "is_account_emptied": is_account_emptied
    }])

    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        col7, col8, col9 = st.columns(3)
        
        with col7:
            st.metric(
                "AI Prediction", 
                "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE",
                delta=f"{prediction_proba[1]*100:.1f}% confidence" if prediction == 1 else f"{prediction_proba[0]*100:.1f}% confidence"
            )
        
        with col8:
            fraud_prob = prediction_proba[1] * 100
            st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
        
        with col9:
            legitimate_prob = prediction_proba[0] * 100
            st.metric("Legitimate Probability", f"{legitimate_prob:.2f}%")
        
        # Detailed results
        st.subheader("📋 Detailed Analysis")
        
        if prediction == 1:
            st.error("🚨 **FRAUD DETECTED!**")
            st.write("""
            **Recommendations:**
            - 🛑 **IMMEDIATE ACTION REQUIRED**: Block this transaction
            - 🔍 Review transaction details thoroughly
            - 📞 Contact account holder for verification
            - 📊 Flag account for enhanced monitoring
            - 🚨 Report to compliance team if confirmed
            """)
        else:
            st.success("✅ **TRANSACTION APPEARS LEGITIMATE**")
            st.write("""
            **Recommendations:**
            - ✅ Transaction can proceed with normal processing
            - 📊 Continue standard monitoring
            - 🔄 No additional verification required
            - 📈 Update risk score if needed
            """)
        
        # Feature importance explanation
        st.subheader("🔍 Model Reasoning")
        st.write("""
        The AI model analyzed the following key factors:
        - **Transaction Type**: Impact of transaction category on fraud risk
        - **Amount Patterns**: Relationship between transaction size and fraud likelihood
        - **Balance Changes**: Analysis of sender/receiver balance modifications
        - **Account Behavior**: Patterns like account emptying and balance ratios
        - **Feature Interactions**: Combined effects of multiple risk factors
        """)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please check that all input values are valid and try again.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🔒 This system is for demonstration purposes only. Always verify predictions with additional security measures.</p>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
</div>
""", unsafe_allow_html=True) 