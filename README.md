# Toxic Text Classification using Transformer

This project implements a Transformer-based model to classify text as toxic or non-toxic. It features a custom Byte Pair Encoding (BPE) tokenizer and a Transformer encoder architecture, built using PyTorch. The model is trained on labeled text data and can predict toxicity in new text samples.

---

## ✨ Features

- **Custom BPE Tokenizer**: Preprocesses text efficiently
- **Transformer Encoder**: Leverages multi-head self-attention
- **Class Imbalance Handling**: Uses weighted loss for better performance
- **Optimized Data Loading**: Multi-worker DataLoader for speed
- **Metrics**: Evaluates with accuracy and weighted F1 score
- **Output**: Saves predictions in TSV format

---

## 📋 Requirements

- Python 3.6+
- PyTorch 1.7+
- tokenizers
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

---
