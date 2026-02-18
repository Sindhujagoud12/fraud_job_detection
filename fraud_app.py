import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix


nltk.download('stopwords')
nltk.download('wordnet')


# load assets
model = joblib.load("fraud_model.pkl")
tfidf = joblib.load("tfidf.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r'http\S+|www\S+','',text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

st.title("🕵️ Job Fraud Detection System")

text = st.text_area("Paste job description")

tele = st.checkbox("Telecommuting")
logo = st.checkbox("Has company logo")
ques = st.checkbox("Has screening questions")
salary = st.checkbox("Salary provided")

if st.button("Check Fraud"):

    cleaned = clean_text(text)
    X_text = tfidf.transform([cleaned])

    meta = [[int(tele), int(logo), int(ques), int(salary)]]
    meta_sparse = csr_matrix(meta)

    X_final = hstack([X_text, meta_sparse])

    prob = model.predict_proba(X_final)[0][1]

    if prob > 0.45:
        st.error(f"🚨 Likely Fraudulent ({prob:.2f})")
    else:
        st.success(f"✅ Likely Genuine ({1-prob:.2f})")
