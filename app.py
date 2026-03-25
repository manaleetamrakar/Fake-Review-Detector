import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake Review Detector")

review = st.text_area("Enter a review")

if st.button("Predict"):
    review_vec = vectorizer.transform([review])
    result = model.predict(review_vec)

    if result[0] == 1:
        st.error("Fake Review ❌")
    else:
        st.success("Genuine Review ✅")