"""
Dataset exploration script for IMDB Sentiment Analysis
Loads the IMDB dataset from HuggingFace and performs initial EDA.
"""

from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

def explore_dataset():
    print("=" * 60)
    print("📦 Loading IMDB Dataset from HuggingFace...")
    print("=" * 60)
    
    dataset = load_dataset("imdb")
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"\n📊 Dataset Structure:")
    print(f"   Train samples: {len(dataset['train']):,}")
    print(f"   Test samples:  {len(dataset['test']):,}")
    
    # Class distribution
    train_labels = dataset['train']['label']
    test_labels = dataset['test']['label']
    
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    label_map = {0: "Negative", 1: "Positive"}
    
    print(f"\n📈 Class Distribution (Train):")
    for label, count in sorted(train_counts.items()):
        print(f"   {label_map[label]}: {count:,} ({count/len(train_labels)*100:.1f}%)")
    
    print(f"\n📈 Class Distribution (Test):")
    for label, count in sorted(test_counts.items()):
        print(f"   {label_map[label]}: {count:,} ({count/len(test_labels)*100:.1f}%)")
    
    # Text length analysis
    train_lengths = [len(text.split()) for text in dataset['train']['text']]
    
    print(f"\n📏 Text Length Stats (Train — word count):")
    print(f"   Min:    {min(train_lengths)}")
    print(f"   Max:    {max(train_lengths)}")
    print(f"   Mean:   {np.mean(train_lengths):.1f}")
    print(f"   Median: {np.median(train_lengths):.1f}")
    print(f"   Std:    {np.std(train_lengths):.1f}")
    
    # Sample examples
    print(f"\n{'=' * 60}")
    print("📝 Sample Reviews:")
    print("=" * 60)
    
    for i in range(3):
        text = dataset['train']['text'][i]
        label = dataset['train']['label'][i]
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\n[{label_map[label]}] (words: {len(text.split())})")
        print(f"   {preview}")
    
    # Save distribution plot
    os.makedirs("data", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Class distribution bar chart
    labels = [label_map[0], label_map[1]]
    train_vals = [train_counts[0], train_counts[1]]
    test_vals = [test_counts[0], test_counts[1]]
    
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, train_vals, width, label='Train', color='#4CAF50')
    axes[0].bar(x + width/2, test_vals, width, label='Test', color='#2196F3')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    
    # Text length histogram
    axes[1].hist(train_lengths, bins=50, color='#FF9800', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Review Length Distribution (Train)')
    axes[1].axvline(np.mean(train_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(train_lengths):.0f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('data/dataset_exploration.png', dpi=150)
    print(f"\n💾 Distribution plot saved to data/dataset_exploration.png")
    
    print(f"\n{'=' * 60}")
    print("✅ Dataset exploration complete!")
    print("=" * 60)
    
    return dataset

if __name__ == "__main__":
    dataset = explore_dataset()
