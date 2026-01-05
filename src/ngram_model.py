import re
from collections import Counter
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

# ---------- Preprocessing ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- Train model ----------
def train_trigram_model(file_path, min_freq=3):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = preprocess_text(raw_text)
    tokens = clean_text.split()

    # ---------- Vocabulary cutoff ----------
    word_counts = Counter(tokens)
    tokens = [
        word if word_counts[word] >= min_freq else "<UNK>"
        for word in tokens
    ]

    N = 3
    train_data, vocab = padded_everygram_pipeline(N, tokens)

    model = Laplace(N)
    model.fit(train_data, vocab)

    return model

