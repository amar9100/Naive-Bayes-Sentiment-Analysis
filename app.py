
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

# 1. Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# 2. Initialize preprocessing objects
stemmer = PorterStemmer()

negative_words = {
    "no", "not", "nor", "never", "neither", "none",
    "nobody", "nothing", "nowhere", "cannot"
}

stop_words = set(ENGLISH_STOP_WORDS) - negative_words


# 3. Load trained models and vectorizers
@st.cache_resource
def load_models():
    bnb_model = joblib.load("models/bernoulli_model.pkl")
    bow_vectorizer = joblib.load("models/bow_vectorizer.pkl")

    mnb_model = joblib.load("models/multinomial_model.pkl")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    return bnb_model, bow_vectorizer, mnb_model, tfidf_vectorizer


bnb_model, bow_vectorizer, mnb_model, tfidf_vectorizer = load_models()


# 4. Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


# 5. Application heading
st.title("😊 Customer Review Sentiment Analysis")

st.write(
    "Enter a customer review to predict whether "
    "the sentiment is Positive or Negative."
)

# 6. Select model
model_choice = st.selectbox(
    "Select Naive Bayes Model",
    [
        "Bernoulli Naive Bayes (BoW)",
        "Multinomial Naive Bayes (TF-IDF)"
    ]
)

# 7. Review input
review = st.text_area(
    "Enter your review",
    placeholder="Example: The food was fantastic!"
)

# 8. Prediction
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        cleaned_review = preprocess_text(review)

        if model_choice == "Bernoulli Naive Bayes (BoW)":
            features = bow_vectorizer.transform([cleaned_review])
            prediction = bnb_model.predict(features)[0]
            probabilities = bnb_model.predict_proba(features)[0]

        else:
            features = tfidf_vectorizer.transform([cleaned_review])
            prediction = mnb_model.predict(features)[0]
            probabilities = mnb_model.predict_proba(features)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probabilities) * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment ☹️")

        st.write(f"**Confidence:** {confidence:.2f}%")
