import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# === Load Data ===
def load_data(labels_path, results_path):
    labels_df = pd.read_csv(labels_path)
    results_df = pd.read_csv(results_path)
    return labels_df, results_df

# === Confusion Matrix ===
def plot_confusion_matrices(results_df, true_col="label"):
    model_cols = {
        "VADER": "vader_sentiment",
        "FinBERT": "finbert_sentiment",
        "Twitter-RoBERTa (Zero-Shot)": "twitter_sentiment",
        "DeBERTa (Fine-Tuned)": "deberta_sentiment",
        "Twitter-RoBERTa (Fine-Tuned)": "finetuned_twitter_sentiment",
        "Hybrid Ensemble": "hybrid_sentiment"
    }
    for name, pred_col in model_cols.items():
        cm = confusion_matrix(results_df[true_col], results_df[pred_col], labels=["negative", "neutral", "positive"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "neutral", "positive"])
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"figures/confusion_matrix_{name.replace(' ', '_').lower()}.png")
        plt.show()
        plt.close()

# === Confidence vs Accuracy ===
def plot_confidence_vs_accuracy(results_df):
    confidence_cols = {
        "Twitter-RoBERTa (Zero-Shot)": "twitter_confidence",
        "DeBERTa (Fine-Tuned)": "finetuned_deberta_sentiment",
        "Twitter-RoBERTa (Fine-Tuned)": "finetuned_twitter_sentiment",
        "Hybrid Ensemble": "hybrid_confidence"
    }
    # Updated prediction columns: use sentiment predictions, not confidence values
    prediction_cols = {
        "Twitter-RoBERTa (Zero-Shot)": "twitter_sentiment",
        "DeBERTa (Fine-Tuned)": "finetuned_deberta_sentiment",
        "Twitter-RoBERTa (Fine-Tuned)": "finetuned_twitter_sentiment",
        "Hybrid Ensemble": "hybrid_sentiment"
    }
    label_col = "label"
    plt.figure(figsize=(10, 6))
    for model, conf_col in confidence_cols.items():
        if conf_col in results_df.columns:
            results_df[conf_col] = pd.to_numeric(results_df[conf_col], errors="coerce") 
            bins = np.linspace(0, 1, 10)
            accuracies = []
            for i in range(len(bins)-1):
                bin_mask = (results_df[conf_col] > bins[i]) & (results_df[conf_col] <= bins[i+1])
                bin_data = results_df[bin_mask]
                if not bin_data.empty:
                    acc = (bin_data[label_col] == bin_data[prediction_cols[model]]).mean()
                    accuracies.append(acc)
                else:
                    accuracies.append(np.nan)
            plt.plot(bins[:-1], accuracies, label=model, marker='o')
    plt.title("Confidence vs Accuracy")
    plt.xlabel("Confidence Bins")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("figures/confidence_vs_accuracy.png")
    plt.close()

# === Per-Class F1 Score Bar Plot ===
def plot_f1_scores(results_df, label_col="label"):
    model_cols = {
        "VADER": "vader_sentiment",
        "FinBERT": "finbert_sentiment",
        "Twitter-RoBERTa (Zero-Shot)": "twitter_sentiment",
        "DeBERTa (Fine-Tuned)": "deberta_sentiment",
        "Twitter-RoBERTa (Fine-Tuned)": "finetuned_twitter_sentiment",
        "Hybrid Ensemble": "hybrid_sentiment"
    }

    data = {}
    for model, pred_col in model_cols.items():
        if pred_col in results_df.columns:
            # Drop rows with NaNs in label or prediction
            valid_data = results_df[[label_col, pred_col]].dropna()
            
            if not valid_data.empty:
                report = classification_report(
                    valid_data[label_col],
                    valid_data[pred_col],
                    output_dict=True
                )
                data[model] = {
                    cls: round(metrics["f1-score"], 3)
                    for cls, metrics in report.items()
                    if cls in ["positive", "neutral", "negative"]
                }

    df = pd.DataFrame(data).T
    df.plot(kind="bar")
    plt.title("Per-Class F1 Score by Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/f1_score_per_class.png")
    plt.show()
    plt.close()

# === Class Distribution Comparison ===
def plot_predicted_distribution(results_df):
    model_cols = ["vader_sentiment", "finbert_sentiment", "twitter_sentiment", "deberta_sentiment", "finetuned_twitter_sentiment", "hybrid_sentiment"]
    true_dist = results_df["label"].value_counts(normalize=True).sort_index()

    plt.figure(figsize=(10, 6))
    width = 0.15
    x = np.arange(len(true_dist.index))

    for i, model in enumerate(model_cols):
        dist = results_df[model].value_counts(normalize=True).reindex(true_dist.index).fillna(0)
        plt.bar(x + i * width, dist.values, width=width, label=model)

    plt.bar(x + len(model_cols) * width, true_dist.values, width=width, label="True Labels", edgecolor="black", linewidth=2)

    plt.xticks(x + width * len(model_cols) / 2, true_dist.index)
    plt.title("Class Distribution (Predicted vs True)")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/class_distribution.png")
    plt.show()
    plt.close()

# === t-SNE of sentence embeddings ===
def plot_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="deep")
    plt.title("t-SNE of Sentence Embeddings")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig("figures/tsne_embeddings.png")
    plt.show()
    plt.close()
    
def get_sentence_embeddings(texts, model_name="ProsusAI/finbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

# === Word Cloud by Sentiment ===
def generate_wordclouds_by_class(results_df):
    classes = results_df['label'].unique()
    for sentiment in classes:
        subset = results_df[results_df['label'] == sentiment]
        text = ' '.join(subset['clean_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for {sentiment.capitalize()} Emails")
        plt.tight_layout()
        plt.savefig(f"figures/wordcloud_{sentiment.lower()}.png")
        plt.show()
        plt.close()
        
# === Word Cloud for Misclassified Texts ===
def wordcloud_misclassified_by_class(results_df, model_col="hybrid_sentiment", label_col="label", text_col="clean_text"):
    mismatches = results_df[results_df[model_col] != results_df[label_col]].dropna(subset=[text_col])
    
    for sentiment in ["positive", "neutral", "negative"]:
        subset = mismatches[mismatches[label_col] == sentiment]
        text = ' '.join(subset[text_col])
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Misclassified Word Cloud (True: {sentiment})")
            plt.tight_layout()
            plt.savefig(f"figures/misclassified_wordcloud_true_{sentiment}.png")
            plt.show()
            plt.close()

if __name__ == "__main__":
    labels_csv = "data/labels.csv"
    results_csv = "results/results.csv"

    labels_df, results_df = load_data(labels_csv, results_csv)

    plot_confusion_matrices(results_df)
    plot_confidence_vs_accuracy(results_df)
    plot_f1_scores(results_df)
    plot_predicted_distribution(results_df)
    generate_wordclouds_by_class(results_df)
    wordcloud_misclassified_by_class(results_df)

    filtered_df = results_df[results_df["clean_text"].notnull()]
    texts = filtered_df["clean_text"].tolist()
    labels = filtered_df["label"].tolist()

    embeddings = get_sentence_embeddings(texts, model_name="yiyanghkust/finbert-tone")

    plot_tsne(embeddings, labels)
