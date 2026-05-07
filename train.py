"""
BERT Fine-tuning script for IMDB Sentiment Analysis
Fine-tunes distilbert-base-uncased on the IMDB dataset.
Using DistilBERT as recommended for beginners — lighter, faster, still great results.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import os
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
SAVE_DIR = "model"
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

print(f"🖥️  Using device: {DEVICE}")


def tokenize_dataset(dataset, tokenizer):
    """Tokenize the entire dataset."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / (step + 1)
            accuracy = correct / total * 100
            print(f"   Step {step+1}/{len(dataloader)} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%")

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def plot_training_history(history, save_path):
    """Plot and save training metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], 'o-', color='#FF5722', label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], 'o-', color='#2196F3', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], 'o-', color='#FF5722', label='Train Acc')
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], 'o-', color='#2196F3', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training plot saved to {save_path}")


def main():
    print("=" * 60)
    print("🚀 BERT Sentiment Analysis — Training Pipeline")
    print("=" * 60)

    # ─── Load Dataset ───
    print("\n📦 Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Create train/val split (80/20 from train set)
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    test_dataset = dataset["test"]

    print(f"   Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # ─── Tokenize ───
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    print("   Tokenizing datasets...")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer)
    test_tokenized = tokenize_dataset(test_dataset, tokenizer)

    train_loader = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_tokenized, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_tokenized, batch_size=BATCH_SIZE)

    # ─── Load Model ───
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable:    {trainable_params:,}")

    # ─── Optimizer & Scheduler ───
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    # ─── Training Loop ───
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0

    print(f"\n{'=' * 60}")
    print(f"🏋️ Training for {EPOCHS} epochs...")
    print(f"{'=' * 60}")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📌 Epoch {epoch}/{EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE, epoch)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, DEVICE)
        val_acc = accuracy_score(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\n   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"   💾 Best model saved! (Val Acc: {val_acc*100:.2f}%)")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training completed in {elapsed/60:.1f} minutes")

    # ─── Final Evaluation on Test Set ───
    print(f"\n{'=' * 60}")
    print("📋 Final Evaluation on Test Set")
    print("=" * 60)

    # Reload best model
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.to(DEVICE)

    test_loss, test_preds, test_labels = evaluate(model, test_loader, DEVICE)

    accuracy = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='binary')
    precision = precision_score(test_labels, test_preds, average='binary')
    recall = recall_score(test_labels, test_preds, average='binary')

    print(f"\n   Accuracy:  {accuracy*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")

    print(f"\n📝 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["Negative", "Positive"]))

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"📊 Confusion Matrix:")
    print(f"   {cm}")

    # Save metrics
    metrics = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "test_accuracy": round(accuracy * 100, 2),
        "test_f1": round(f1 * 100, 2),
        "test_precision": round(precision * 100, 2),
        "test_recall": round(recall * 100, 2),
        "training_time_minutes": round(elapsed / 60, 1),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to {metrics_path}")

    # Plot training history
    plot_training_history(history, os.path.join(SAVE_DIR, "training_history.png"))

    print(f"\n{'=' * 60}")
    print("✅ All done! Model ready for deployment.")
    print(f"   Model saved in: {SAVE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
