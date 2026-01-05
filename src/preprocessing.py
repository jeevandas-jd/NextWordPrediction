import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text)      # normalize spaces
    return text.strip()

with open("data/alice.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

clean_text = preprocess_text(raw_text)
tokens = clean_text.split()

print(tokens[:20])
print("Total tokens:", len(tokens))
