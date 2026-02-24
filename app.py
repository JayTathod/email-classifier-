import streamlit as st
import pickle

model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("Spam Email Classifier")

input_text = st.text_area("Enter email text")

if st.button("Predict"):
    vector = vectorizer.transform([input_text])
    result = model.predict(vector)[0]

    if result == 1:
        st.error("🚨 This email is SPAM")
    else:
        st.success("✅ This email is NOT Spam")