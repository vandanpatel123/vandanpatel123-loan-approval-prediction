import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('loan_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App configuration
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💸")

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=60)
st.sidebar.title("📊 About")
st.sidebar.info("""
This app predicts loan approval based on applicant's financial & personal details.

Built with ❤️ using **Streamlit** and **Machine Learning**.
""")

# Title & description
st.title("🏦 Loan Approval Prediction")
st.markdown("Predict whether your loan will be **approved or rejected** based on your details.")

# Input fields
with st.form("prediction_form"):
    st.subheader("Enter Applicant Details:")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income (Monthly)", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income (Monthly)", min_value=0)
    loan_amount = st.number_input("Loan Amount (in ₹ thousands)", min_value=0.0)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0.0)
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit = st.form_submit_button("🔍 Predict")

# Convert input to DataFrame
def preprocess_input():
    return pd.DataFrame([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        3 if dependents == "3+" else int(dependents),
        0 if education == "Graduate" else 1,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        1.0 if credit_history == "Good (1)" else 0.0,
        {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    ]], columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ])

# Show prediction
if submit:
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction] if hasattr(model, "predict_proba") else None

    result = "✅ **Loan Approved**" if prediction == 1 else "❌ **Loan Rejected**"
    st.markdown("### Prediction Result:")
    st.success(result)

    if probability is not None:
        st.markdown(f"**Confidence:** {round(probability * 100, 2)}%")

    st.markdown("---")

