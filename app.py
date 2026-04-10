import streamlit as st
import pickle

# ------------------ Load Models ------------------
lr = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
svd = pickle.load(open("svd.pkl", "rb"))

# ------------------ Text Cleaning Function ------------------
# (Make sure this matches your training preprocessing)
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
def clean_text(text):
  text = text.lower()
  text = re.sub(r"<.*?>","",text) # remove html
  text = re.sub(r"http\S+", "", text) # remove url
  text = re.sub(r"[^a-zA-Z\s]","",text) # remove special character
  words = text.split()
  words = [w for w in words if w not in stop_words]
  return " ".join(words)

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown('<div class="title">💬 Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze the sentiment of your text instantly</div>', unsafe_allow_html=True)

# ------------------ Input ------------------
user_input = st.text_area("Enter your review:", height=150, placeholder="Type your review here...")

# ------------------ Prediction ------------------
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)

        review_vec = tfidf.transform([cleaned_input])
        review_vec_reduced = svd.transform(review_vec)
        prediction = lr.predict(review_vec_reduced)[0]

        label_map = {
            0: "Negative 😡",
            1: "Neutral 😐",
            2: "Positive 😊"
        }

        sentiment = label_map[int(prediction)]

        # Color mapping
        color_map = {
            0: "#e74c3c",
            1: "#f1c40f",
            2: "#2ecc71"
        }

        # Display Result
        st.markdown(
            f'<div class="result-box" style="background-color:{color_map[int(prediction)]}; color:white;">'
            f'Predicted Sentiment: {sentiment}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Optional: Confidence Score
        if hasattr(lr, "predict_proba"):
            proba = lr.predict_proba(review_vec_reduced)[0]
            st.write("### Confidence Scores:")
            st.write({
                "Negative": round(proba[0], 3),
                "Neutral": round(proba[1], 3),
                "Positive": round(proba[2], 3)
            })