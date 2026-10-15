import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once; safe to call multiple times
for pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)          # emails
    text = re.sub(r"\d+", " ", text)              # digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [LEMM.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)