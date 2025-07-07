import streamlit as st
import pandas as pd
import os
import base64
from config import FINETUNED_CHECKPOINT, BERT_CHECKPOINT, MAX_LEN, MAPPING
from utils import clean_text, preprocess, classify
from transformers import AutoModelForSequenceClassification, BertTokenizer

def set_video_background(video_url):
    """Set a video background using an external link."""
    video_html = f"""
    <style>
    .stApp {{
        position: relative;
        height: 100vh;
        overflow: hidden;
    }}
    video {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover; !important
        z-index: -1; !important
    }}
    </style>
    
    <video autoplay loop muted playsinline>
        <source src="{video_url}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# Use a publicly accessible video URL
set_video_background("https://drive.google.com/uc?export=download&id=1tZE24F6_kChA4Rk5uRHgegL6TFwc3hLE")

st.markdown("# SkipOrStick : Netflix ver 🎬")
st.write("Welcome! Analyze Netflix show reviews here.")

# Load Netflix titles data
netflix_data = pd.read_csv("../data/netflix_titles.csv")
titles = set(netflix_data['title'].str.lower())  # Convert to lowercase for case-insensitive matching

# Path to store reviews
REVIEWS_FILE = "reviews.csv"

# Load past reviews if the file exists
if os.path.exists(REVIEWS_FILE):
    reviews_df = pd.read_csv(REVIEWS_FILE)
else:
    reviews_df = pd.DataFrame(columns=["title", "sentiment"])

@st.cache(allow_output_mutation=True)
def load_model_tokenizer(model_checkpoint, tokenizer_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_checkpoint)
    return model, tokenizer

model, tokenizer = load_model_tokenizer(FINETUNED_CHECKPOINT, BERT_CHECKPOINT)

# Input: Show Title
show_title = st.text_input("Enter the Netflix show title").strip().lower()

if show_title:
    if show_title in titles:
        st.success(f"✅ '{show_title.title()}' is a Netflix show! You can proceed with the review.")

        # Input: Review
        review = st.text_area("Write a movie review and hit the process button to classify it")

        if st.button('Process'):
            if review.isspace() or len(review) == 0:
                st.markdown('**No review provided!** Please enter some text and then hit Process')
            else:
                preprocessed_review = preprocess(review, tokenizer=tokenizer, max_len=MAX_LEN, clean_text=clean_text)
                out = classify(inputs=preprocessed_review, model=model, mapping=MAPPING)

                sentiment = out['Label']
                confidence = out['Confidence'] * 100
                sentiment_score = 1 if sentiment == 'positive' else 0

                # Save review to DataFrame
                new_review = pd.DataFrame({"title": [show_title], "sentiment": [sentiment_score]})
                reviews_df = pd.concat([reviews_df, new_review], ignore_index=True)

                # Save to CSV
                reviews_df.to_csv(REVIEWS_FILE, index=False)

                thumb = ":smiley:" if sentiment == 'positive' else ":angry:"
                st.markdown(f"The review is **{sentiment}** {thumb} with a confidence of **{confidence:.2f}%**.")

        # Calculate overall prediction from **all** reviews of the show
        show_reviews = reviews_df[reviews_df["title"] == show_title]
        
        if not show_reviews.empty:
            avg_sentiment = show_reviews["sentiment"].mean()  # Compute **average** sentiment
            review_count = len(show_reviews)

            # General recommendation
            prediction = "Recommended 👍" if avg_sentiment > 0.5 else "Not Recommended 👎"
            st.markdown(f"### Prediction for **{show_title.title()}**: {prediction} (Based on {review_count} reviews, avg score: {avg_sentiment:.2f})")

            # **NEW**: Predict show success if reviews exceed 50
            if review_count > 50:
                if avg_sentiment < 0.5:
                    success_prediction = "Miss ❌"
                elif 0.5 <= avg_sentiment <= 0.8:
                    success_prediction = "Likely a Hit 🤔"
                else:
                    success_prediction = "Hit 🎉"

                st.markdown(f"### Success Prediction for **{show_title.title()}**: {success_prediction}")

    else:
        st.error(f"❌ '{show_title.title()}' is NOT a Netflix show. Please enter a valid Netflix title.")
