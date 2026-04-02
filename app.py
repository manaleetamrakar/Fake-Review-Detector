import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake Review Detector", layout="centered")

# Custom styling (JoinArena inspired)
st.markdown("""
    <style>
    body {
        background-color: #f5f7fb;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
    }
    h1 {
        color: #0b1c3d;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #2b5cff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title + description
st.title("Fake Review Detector")
st.write("Analyze product reviews and classify them as genuine or fake using a machine learning model.")

# Input
review = st.text_area("Enter a product review")

# Button
if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vec = vectorizer.transform([review])
        result = model.predict(review_vec)
        proba = model.predict_proba(review_vec)

        if result[0] == 1:
            st.error(f"Fake review (confidence: {proba[0][1]*100:.2f}%)")
        else:
            st.success(f"Genuine review (confidence: {proba[0][0]*100:.2f}%)")

# Divider
st.markdown("---")

# Examples
st.subheader("Example inputs")

st.code("This product is amazing. Best purchase I have made.")
st.code("The product works fine but delivery was delayed.")
