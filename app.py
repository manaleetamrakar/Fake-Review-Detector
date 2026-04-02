import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️")

st.title("🕵️ Fake Review Detector")
st.write("Enter a product review below to check if it is genuine or fake.")

# Input box
review = st.text_area("✍️ Enter your review here:")

# Predict button
if st.button("🔍 Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        review_vec = vectorizer.transform([review])
        result = model.predict(review_vec)
        proba = model.predict_proba(review_vec)

        if result[0] == 1:
            st.error(f"🚨 Fake Review Detected! (Confidence: {proba[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ Genuine Review (Confidence: {proba[0][0]*100:.2f}%)")

# Example section
st.markdown("---")
st.subheader("💡 Try these examples:")

st.code("This product is AMAZING!!! Best purchase ever!!!")
st.code("The product quality is decent but delivery was slow.")
