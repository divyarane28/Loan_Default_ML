import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load('loan_default_model.pkl')

st.title("Loan Default Prediction App")

st.write("Enter details below:")

# inputs (you can adjust based on your dataset)

person_age = st.number_input("Age", min_value=0, step=1, format="%d")
person_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
person_education = st.selectbox("Education Level",["High School", "Bachelor", "Master", "Doctorate", "Associate"])
person_income = st.number_input("Person Income", min_value=0, step=1, format="%d")
person_emp_exp = st.number_input("Person Experience (years)", min_value=0, step=1, format="%d")
person_home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE"])
loan_amnt = st.number_input("Loan Amount", min_value=0, step=1, format="%d")
loan_intent = st.selectbox("Loan Type",["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Interest Rate", min_value=0, step=1, format="%d")
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, step=1, format="%d")
credit_score = st.number_input("Credit Score", min_value=0, step=1, format="%d")
# 🔹 Encoding maps
gender_map = {"Male": 0, "Female": 1, "Other": 2}
education_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "Doctorate": 3,
    "Associate": 4
}
home_map = {"OWN": 0, "RENT": 1, "MORTGAGE": 2}
loan_intent_map = {
    "PERSONAL": 0,
    "EDUCATION": 1,
    "MEDICAL": 2,
    "VENTURE": 3,
    "HOMEIMPROVEMENT": 4,
    "DEBTCONSOLIDATION": 5
}
if person_income > 0:
    loan_percent_income = loan_amnt / person_income
else:
    st.error("Income must be greater than 0")
    st.stop()
# simple predict button
if st.button("Predict"):

    input_data = np.array([[
        person_age,
        gender_map[person_gender],
        education_map[person_education],
        person_income,
        person_emp_exp,
        home_map[person_home_ownership],
        loan_amnt,
        loan_intent_map[loan_intent],
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score
    ]]).astype(int)

    prediction = model.predict(input_data)

    prob = model.predict_proba(input_data)[0][1]
    st.write(f"Default Probability: {round(prob*100,2)}%")

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Customer may default")
    else:
        st.success("✅ Low Risk: Customer is safe")