from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import nltk

nltk.download("punkt", quiet=True)

MODEL_PATH = r"D:/1.SHREEE/Summarization+NER/utils/distilbert_ner_model"

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# Create pipeline with aggregation
ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def ner_detection(texts)    :
    """
    texts: str or list of str
    returns: list of entity dicts with sentence number instead of full sentence
    """
    if isinstance(texts, str):
        texts = [texts]

    results = []
    for text in texts:
        sentences = nltk.sent_tokenize(text)

        for idx, sent in enumerate(sentences, start=1):
            entities = ner_pipeline(sent)
            for e in entities:
                results.append({
                    "sentence_num": idx,
                    "word": e["word"],
                    "entity_group": e["entity_group"],
                    "score": round(float(e["score"]), 4)
                })
    return results

# Example run
if __name__ == "__main__":
    sample_text = "Neha met Arjun at Starbucks. Arjun came back from London."
    output = ner_detection(sample_text)
    for row in output:
        print(row)
