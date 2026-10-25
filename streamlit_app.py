import re
import html
import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Research Abstract Classifier")


@st.cache_resource
def load_everything():
    # runs once, then Streamlit remembers the result
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    tfidf = joblib.load("tfidf.joblib")
    model = joblib.load("model.joblib")        # Logistic Regression
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return tfidf, model, stop_words, lemmatizer


tfidf, model, stop_words, lemmatizer = load_everything()


def clean(text):
    # same steps, same order as the notebook
    text = str(text).lower()                                   # lowercase
    text = html.unescape(text)                                 # &amp; -> &
    text = re.sub(r"<.*?>", " ", text)                         # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)              # URLs
    text = re.sub(r"[^\w\s]", " ", text)                       # punctuation
    text = re.sub(r"\s+", " ", text).strip()                   # extra spaces
    words = [w for w in text.split() if w not in stop_words]   # stopwords
    tokens = word_tokenize(" ".join(words), preserve_line=True)    # tokenize
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)       # lemmatize


st.title("Research Abstract Classifier")
st.write("Paste an abstract and the model guesses its subject: "
         "**Science**, **Business** or **Social Science**. English abstracts only.")

tab1, tab2 = st.tabs(["Paste an abstract", "Upload a CSV"])

with tab1:
    abstract = st.text_area("Abstract", height=220)
    if st.button("Predict"):
        if len(abstract.split()) < 15:
            st.warning("Please paste a longer abstract (at least 15 words).")
        else:
            x = tfidf.transform([clean(abstract)])
            probs = model.predict_proba(x)[0]
            result = pd.Series(probs, index=model.classes_).sort_values(ascending=False)
            st.success(f"Predicted subject: {result.index[0]} ({result.iloc[0]:.0%} sure)")
            st.bar_chart(result)

with tab2:
    st.write("Upload a CSV file that has a column named `abstract`.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        if "abstract" not in df.columns:
            st.error("The CSV needs a column named 'abstract'.")
        else:
            X = tfidf.transform(df["abstract"].fillna("").apply(clean))
            df["predicted_label"] = model.predict(X)
            df["confidence"] = model.predict_proba(X).max(axis=1).round(3)
            st.dataframe(df.head(20))
            st.download_button("Download all predictions",
                               df.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv",
                               mime="text/csv")