"""Text cleaning utilities for the spam classifier."""

import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    """Download a small NLTK resource only when it is missing."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


@lru_cache(maxsize=1)
def _text_tools() -> tuple[set[str], WordNetLemmatizer]:
    _ensure_nltk_resource("corpora/stopwords", "stopwords")
    _ensure_nltk_resource("corpora/wordnet", "wordnet")
    _ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    return set(stopwords.words("english")), WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Normalize SMS text for TF-IDF feature extraction."""
    if not isinstance(text, str):
        return ""

    stop_words, lemmatizer = _text_tools()

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        lemmatizer.lemmatize(token)
        for token in text.split()
        if token and token not in stop_words
    ]
    return " ".join(tokens)
