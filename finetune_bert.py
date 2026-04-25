import os
import re
import pandas as pd
import numpy as np
import torch
import nltk
from datasets import Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse

# === Setup ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# === Preprocessing Setup ===
stop_words = set(stopwords.words('english')).union({'ect', 'hou', 'enron', 'subject', 'corp'})
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'-{2,}.*Original Message.*-{2,}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Regards,|Best regards,|Thanks,|Sent from)', '', text)
    text = re.sub(r'\S+@\S+', 'emailaddr', text)
    text = re.sub(r'http\S+|www\.\S+', 'url', text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'\d+', '', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

def print_summary(model_name, best_epoch, train_acc, train_loss, test_acc, test_loss, y_true, y_pred, target_names):
    print(f"\n{model_name} Model:")
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Accuracy at Best Epoch: {train_acc:.4f}")
    print(f"Training Loss at Best Epoch: {train_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("=" * 60)
    print(f"{model_name} Model Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=2))
    
def print_summary(model_name, best_epoch, train_acc, train_loss, test_acc, test_loss, y_true, y_pred, target_names, save_path=None):
    summary = []
    summary.append(f"\n{model_name} Model:")
    summary.append(f"Best Epoch: {best_epoch}")
    summary.append(f"Training Accuracy at Best Epoch: {train_acc:.4f}")
    summary.append(f"Training Loss at Best Epoch: {train_loss:.4f}")
    summary.append(f"Test Accuracy: {test_acc:.4f}")
    summary.append(f"Test Loss: {test_loss:.4f}")
    summary.append("=" * 60)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=2)
    summary.append(f"{model_name} Model Classification Report:")
    summary.append(report)

    # Print to console
    print("\n".join(summary))

    # Save to file if path is provided
    if save_path:
        with open(save_path, "w") as f:
            f.write("\n".join(summary))
        print(f"Saved training summary to {save_path}")
        

def create_optimizer(model, training_args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

# === Load & Prepare ===
def load_data(path):
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean_text)
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["label"].map(label_map)
    return df

# === Train with Stratified 5-Fold CV and Dynamic Rebalancing ===
def train_model_with_cv(model_name, save_dir_base, df, num_epochs=4, max_per_class=1500):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\nFold {fold + 1}/5")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        capped_dfs = []
        for lbl in [0, 1, 2]:
            subset = train_df[train_df["label"] == lbl]
            if len(subset) > max_per_class:
                subset = subset.sample(max_per_class, random_state=42)
            capped_dfs.append(subset)

        train_df = pd.concat(capped_dfs)

        label_counts = train_df["label"].value_counts()
        max_count = label_counts.max()
        neutral_count = label_counts.get(1, 0)

        if neutral_count < max_count:
            neutral_df = train_df[train_df["label"] == 1]
            upsampled = neutral_df.sample(max_count - neutral_count, replace=True, random_state=42)
            train_df = pd.concat([train_df, upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

        # === Tokenization ===
        def tokenize(example):
            return tokenizer(example["clean_text"], padding="max_length", truncation=True, max_length=256)

        train_ds = Dataset.from_pandas(train_df[["clean_text", "label"]]).map(tokenize, batched=True)
        val_ds = Dataset.from_pandas(val_df[["clean_text", "label"]]).map(tokenize, batched=True)
        train_ds = train_ds.remove_columns(["clean_text"])
        val_ds = val_ds.remove_columns(["clean_text"])
        train_ds.set_format("torch")
        val_ds.set_format("torch")

        class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_df["label"])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

        class WeightedModel(nn.Module):
            def __init__(self, model, weights):
                super().__init__()
                self.model = model
                self.loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if labels is not None:
                    labels = labels.to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.loss_fn(logits, labels) if labels is not None else None
                return {"loss": loss, "logits": logits}

        model = WeightedModel(base_model, class_weights_tensor)

        training_args = TrainingArguments(
            output_dir=f"{save_dir_base}/fold{fold+1}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            logging_steps=10,
            learning_rate=1.5e-5,
            max_grad_norm=1.0,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=f"{save_dir_base}/logs_fold{fold+1}",
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
        )
        
        optimizer = create_optimizer(model, training_args)
        num_training_steps = int(len(train_ds) / training_args.per_device_train_batch_size) * training_args.num_train_epochs
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(training_args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            optimizers=(optimizer, lr_scheduler)
        )

        trainer.train()

        train_metrics = trainer.evaluate(train_ds)
        results = trainer.predict(val_ds)
        y_pred = np.argmax(results.predictions, axis=-1)
        y_true = results.label_ids
        
        summary_path = os.path.join(save_dir_base, f"fold{fold+1}", "training_summary.txt")

        print_summary(
            model_name=model_name.split("/")[-1].capitalize(),
            best_epoch=num_epochs,
            train_loss=train_metrics.get('eval_loss', 0.0),
            test_loss=results.metrics.get('test_loss', 0.0),
            test_acc=results.metrics.get('test_accuracy', 0.0),
            train_acc=1.0 - train_metrics.get('eval_loss', 0.0),
            y_true=y_true,
            y_pred=y_pred,
            target_names=["negative", "neutral", "positive"],
            save_path=summary_path
        )

        base_model.save_pretrained(f"{save_dir_base}/fold{fold+1}")
        tokenizer.save_pretrained(f"{save_dir_base}/fold{fold+1}")
        print(f"Saved fold {fold+1} model and tokenizer to {save_dir_base}/fold{fold+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/labels.csv", help="CSV with 'text' and 'label'")
    args = parser.parse_args()

    df = load_data(args.data)
    train_model_with_cv("microsoft/deberta-v3-base", "./finetuned_deberta", df, num_epochs=4)
    train_model_with_cv("cardiffnlp/twitter-roberta-base-sentiment", "./finetuned_twitter", df, num_epochs=3)
