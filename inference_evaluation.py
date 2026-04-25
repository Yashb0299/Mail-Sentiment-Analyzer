import os
import re
import pandas as pd
import nltk
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.metrics import classification_report
from tqdm import tqdm
from math import ceil

# === Setup ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
device = 0 if torch.backends.mps.is_available() else -1
print("Device set to:", "MPS" if device == 0 else "CPU")

stop_words = set(stopwords.words('english')).union({'ect', 'hou', 'enron', 'subject', 'corp'})
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    # Remove forwarded/quoted lines
    text = re.sub(r'-{2,}.*Original Message.*-{2,}', '', text, flags=re.IGNORECASE)
    # Remove common email closings
    text = re.sub(r'(Regards,|Best regards,|Thanks,|Sent from)', '', text)
    # Replace emails/URLs
    text = re.sub(r'\S+@\S+', 'emailaddr', text)
    text = re.sub(r'http\S+|www.\S+', 'url', text)
    # Remove non-ascii and lower
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'\d+', '', text.lower())
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize & lemmatize, remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

def map_twitter_label(label):
==
    mapping = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
    return mapping.get(label, label)

def majority_vote(predictions):
    # Return most common predicted label
    return Counter(predictions).most_common(1)[0][0]
    
def get_vader_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive', score
    elif score <= -0.05:
        return 'negative', abs(score)
    else:
        return 'neutral', abs(score)

# === Load Manually Labeled Data ===
df = pd.read_csv("data/labels.csv")  # Ensure it has "text" and "label"
df["clean_text"] = df["text"].apply(clean_text)

# === VADER Sentiment ===
print("Running VADER sentiment...\n")
vader_results = df['clean_text'].apply(get_vader_sentiment)
df['vader_sentiment'] = vader_results.apply(lambda x: x[0])
df['vader_confidence'] = vader_results.apply(lambda x: x[1])

# === Raw FinBERT ===
tqdm.write("Running FinBERT sentiment (batched)...")
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", device=device)

texts = df["clean_text"].str[:512].tolist()
batch_size = 32
finbert_outputs = []

for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT", total=ceil(len(texts)/batch_size)):
    batch = texts[i:i+batch_size]
    outputs = finbert(batch, truncation=True, max_length=512)
    finbert_outputs.extend(outputs)

df["finbert_sentiment"] = [out["label"].lower() for out in finbert_outputs]

# === Raw Twitter-RoBERTa ===
tqdm.write("Running Twitter RoBERTa sentiment (batched)...")
twitter = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

twitter_outputs = []
for i in tqdm(range(0, len(texts), batch_size), desc="Twitter RoBERTa", total=ceil(len(texts)/batch_size)):
    batch = texts[i:i+batch_size]
    outputs = twitter(batch, truncation=True, max_length=512)
    twitter_outputs.extend(outputs)

twitter_labels = []
twitter_scores = []
for out in twitter_outputs:
    label = map_twitter_label(out["label"].lower())
    twitter_labels.append(label)
    twitter_scores.append(out["score"])

df["twitter_sentiment"] = twitter_labels
df["twitter_confidence"] = twitter_scores

# === Load 5-Fold Twitter-RoBERTa Fine-Tuned Pipelines ===
twitter_pipelines = []
twitter_dirs = [f"./finetuned_twitter/fold{i}" for i in range(1, 6)]
for fold_dir in twitter_dirs:
    model = AutoModelForSequenceClassification.from_pretrained(fold_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(fold_dir, local_files_only=True)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    twitter_pipelines.append(pipe)

# === Load 5-Fold deBERTa Fine-Tuned Pipelines ===
deberta_pipelines = []
deberta_dirs = [f"./finetuned_deberta/fold{i}" for i in range(1, 6)]
for fold_dir in deberta_dirs:
    model = AutoModelForSequenceClassification.from_pretrained(fold_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(fold_dir, local_files_only=True)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    deberta_pipelines.append(pipe)

# === ID Mapping for Label Index to Sentiment ===
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# === BATCHED 1) Fine-Tuned deBERTa Ensemble ===
def predict_deberta_ensemble_batched(texts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="DeBERTa Ensemble"):
        batch = texts[i:i+batch_size]
        outputs_per_model = [pipe(batch, truncation=True, max_length=512) for pipe in deberta_pipelines]

        for j in range(len(batch)):
            labels = []
            scores = []
            for model_outputs in outputs_per_model:
                out = model_outputs[j]
                lbl = out['label'].lower()
                if 'label_' in lbl:
                    idx = int(lbl.split('_')[-1])
                    lbl = id2label[idx]
                labels.append(lbl)
                scores.append((lbl, out['score']))

            voted_label = majority_vote(labels)
            avg_conf = sum(score for lbl, score in scores if lbl == voted_label) / max(1, sum(1 for lbl, _ in scores if lbl == voted_label))
            results.append((voted_label, avg_conf))
    return results

# === BATCHED 2) Fine-Tuned Twitter-RoBERTa Ensemble ===
def predict_twitter_ensemble_batched(texts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Twitter-RoBERTa Ensemble"):
        batch = texts[i:i+batch_size]
        outputs_per_model = [pipe(batch, truncation=True, max_length=512) for pipe in twitter_pipelines]

        for j in range(len(batch)):
            labels = []
            scores = []
            for model_outputs in outputs_per_model:
                out = model_outputs[j]
                lbl = out['label'].lower()
                if 'label_' in lbl:
                    idx = int(lbl.split('_')[-1])
                    lbl = id2label[idx]
                labels.append(lbl)
                scores.append((lbl, out['score']))

            voted_label = majority_vote(labels)
            avg_conf = sum(score for lbl, score in scores if lbl == voted_label) / max(1, sum(1 for lbl, _ in scores if lbl == voted_label))
            results.append((voted_label, avg_conf))
    return results

# === BATCHED 3 & 4) Hybrid Soft-Voting Ensemble ===
def get_probs_batched(pipelines, texts, label_order=["label_0", "label_1", "label_2"], batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        logits_sums = [torch.zeros(len(label_order)) for _ in batch]

        for pipe in pipelines:
            model_outputs = pipe(batch, truncation=True, top_k=None, max_length=512)
            for j, output in enumerate(model_outputs):
                score_map = {s['label'].lower(): s['score'] for s in output}
                logits = torch.tensor([score_map.get(lbl, 0.0) for lbl in label_order])
                logits_sums[j] += logits

        averaged_probs = [logits / len(pipelines) for logits in logits_sums]
        all_probs.extend(averaged_probs)
    return all_probs

def predict_hybrid_softvote_batched(texts, weight_twitter=0.8, weight_deberta=0.2, batch_size=32):
    twitter_probs = get_probs_batched(twitter_pipelines, texts, batch_size=batch_size)
    deberta_probs = get_probs_batched(deberta_pipelines, texts, batch_size=batch_size)
    hybrid_results = []
    for tw, db in zip(twitter_probs, deberta_probs):
        combined = weight_twitter * tw + weight_deberta * db
        pred_idx = torch.argmax(combined).item()
        hybrid_results.append((id2label[pred_idx], combined[pred_idx].item()))
    return hybrid_results

# === Apply Predictions (Batched Version) ===

# 1) deBERTa ensemble (batched)
print("Running fine-tuned DeBERTa ensemble sentiment (batched)...")
deberta_results = predict_deberta_ensemble_batched(texts)
df["deberta_sentiment"] = [x[0] for x in deberta_results]
df["deberta_confidence"] = [x[1] for x in deberta_results]

# 2) Twitter-RoBERTa ensemble (batched)
print("Running fine-tuned Twitter-RoBERTa ensemble sentiment (batched)...")
twi_results = predict_twitter_ensemble_batched(texts)
df["finetuned_twitter_sentiment"] = [x[0] for x in twi_results]
df["finetuned_twitter_confidence"] = [x[1] for x in twi_results]

# 3) Hybrid ensemble (batched)
print("Running hybrid ensemble sentiment (batched)...")
hybrid_results = predict_hybrid_softvote_batched(texts)
df["hybrid_sentiment"] = [x[0] for x in hybrid_results]
df["hybrid_confidence"] = [x[1] for x in hybrid_results]

# === Emotion Classification (batched)
print("Running emotion classification (batched)...")
emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=device)
emotion_outputs = list(tqdm(emotion(texts, batch_size=32, truncation=True, max_length=512), desc="Emotion"))
df["emotion"] = [out[0]["label"] for out in emotion_outputs]
df["emotion_confidence"] = [out[0]["score"] for out in emotion_outputs]

# === Evaluation Against Manual Labels ===

print("VADER Sentiment:")
print(classification_report(df["label"], df["vader_sentiment"], digits=3))

print("\n--- Evaluation Against Manual Labels ---")
print("FinBERT (raw):")
print(classification_report(df["label"], df["finbert_sentiment"], digits=3))

print("Twitter RoBERTa (raw):")
print(classification_report(df["label"], df["twitter_sentiment"], digits=3))

print("Fine-tuned DeBERTa Ensemble:")
print(classification_report(df["label"], df["deberta_sentiment"], digits=3))

print("Fine-tuned Twitter-RoBERTa Ensemble:")
print(classification_report(df["label"], df["finetuned_twitter_sentiment"], digits=3))

print("Hybrid Ensemble (deBERTa + Twitter-RoBERTa):")
print(classification_report(df["label"], df["hybrid_sentiment"], digits=3))

# === Sample Output (based on stored predictions) ===
rand_idx = df.sample(1).index[0]
print("\nTrue Label:", df.loc[rand_idx, 'label'])
print("VADER:", df.loc[rand_idx, 'vader_sentiment'],
      "| Score:", round(df.loc[rand_idx, 'vader_confidence'], 3))
print("FinBERT:", df.loc[rand_idx, 'finbert_sentiment'],
      "| (precomputed)")
print("Twitter:", df.loc[rand_idx, 'twitter_sentiment'],
      "| Conf:", round(df.loc[rand_idx, 'twitter_confidence'], 3))
print("Fine-tuned DeBERTa:", df.loc[rand_idx, 'deberta_sentiment'], 
      "| Avg Conf:", round(df.loc[rand_idx, 'deberta_confidence'], 3))
print("Fine-tuned Twitter:", df.loc[rand_idx, 'finetuned_twitter_sentiment'],
      "| Avg Conf:", round(df.loc[rand_idx, 'finetuned_twitter_confidence'], 3))
print("Hybrid:", df.loc[rand_idx, 'hybrid_sentiment'],
      "| Avg Conf:", round(df.loc[rand_idx, 'hybrid_confidence'], 3))
print("Emotion:", df.loc[rand_idx, 'emotion'],
      "| Conf:", round(df.loc[rand_idx, 'emotion_confidence'], 3))

# === Save All Results to CSV ===
df.to_csv("results/results.csv", index=False)
print("\nSaved full predictions to results/results.csv")
