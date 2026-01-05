from ngram_model import train_trigram_model

def predict_next_word(model, context):
    context = tuple(context.lower().split())
    candidates = []

    for word in model.vocab:
        if word in ("<s>", "</s>", "<UNK>"):
            continue
        if len(word) <= 1:
            continue

        prob = model.score(word, context)
        candidates.append((word, prob))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None


if __name__ == "__main__":
    model = train_trigram_model("data/alice.txt")

    context = "alice was"
    prediction = predict_next_word(model, context)

    print(f"Next word prediction for '{context}':")
    print(prediction)

