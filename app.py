import streamlit as st
import joblib

# load saved model
model = joblib.load("spam_model.pkl")

st.title("Spam Email Classifier")
st.write("Enter the email text below to check whether it is spam or legitimate.")

email_text = st.text_area("Enter email text")

if st.button("Predict"):
    prediction = model.predict([email_text])[0]
    probability = model.predict_proba([email_text]).max()

    if prediction == 1:
        st.error("Spam Email")
    else:
        st.success("Legitimate Email")

    st.write("Confidence Score:", round(probability * 100, 2), "%")
