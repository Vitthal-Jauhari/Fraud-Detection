import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load(
    r"C:\Codes\fraud_detection\Fraud_Detection_Pipeline.pkl"
)

# ---------------- UI ---------------- #

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Fraud Detection App")

st.markdown(
    "Enter transaction details and click Predict."
)

st.divider()

# EXACT categories from trained encoder
transaction_type = st.selectbox(
    "Transaction Type",
    [
        'CASH_IN',
        'CASH_OUT',
        'DEBIT',
        'PAYMENT',
        'TRANSFER'
    ]
)

amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=1000.0
)

oldbalanceOrg = st.number_input(
    "Old Balance [Sender]",
    min_value=0.0,
    value=10000.0
)

newbalanceOrig = st.number_input(
    "New Balance [Sender]",
    min_value=0.0,
    value=9000.0
)

oldbalanceDest = st.number_input(
    "Old Balance [Receiver]",
    min_value=0.0,
    value=10000.0
)

newbalanceDest = st.number_input(
    "New Balance [Receiver]",
    min_value=0.0,
    value=11000.0
)

# ---------------- Prediction ---------------- #

if st.button("Predict Fraud"):

    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    prediction = model.predict(input_data)[0]

    # Optional probability
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader(f"Prediction: {int(prediction)}")

    st.write(f"Fraud Probability: {probability:.2%}")

    if prediction == 1:
        st.error("⚠️ This transaction might be fraudulent.")
    else:
        st.success("✅ Transaction looks legitimate.")