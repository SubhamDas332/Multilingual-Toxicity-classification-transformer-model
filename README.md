# Toxic Text Classification using Transformer

This project implements a Transformer-based model to classify text as toxic or non-toxic. It features a custom Byte Pair Encoding (BPE) tokenizer and a Transformer encoder architecture, built using PyTorch. The model is trained on labeled text data and can predict toxicity in new text samples.

---

##  Features

- **Custom BPE Tokenizer**: Preprocesses text efficiently
- **Transformer Encoder**: Leverages multi-head self-attention
- **Class Imbalance Handling**: Uses weighted loss for better performance
- **Optimized Data Loading**: Multi-worker DataLoader for speed
- **Metrics**: Evaluates with accuracy and weighted F1 score
- **Output**: Saves predictions in TSV format

---

##  Requirements

- Python 3.6+
- PyTorch 1.7+
- tokenizers
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

---

Making Predictions
After training:

Loads best_transformer_model.pth
Processes test.tsv
Saves predictions to predictions.tsv (column: predicted, 0 or 1)
Note: Fix any RuntimeError by ensuring the saved model matches the architecture (e.g., embed_size=128).

## Results

The following table compares the performance of the Language aware Model and the Baseline Model across different languages and metrics.

**Table 1: Comparison of Language aware Model and Baseline Model**

| Language       | Metric    | Language aware Model | Baseline Model |
|:---------------|:----------|---------------------:|---------------:|
| Multilingual   | F1 Score  | 0.7041               | 0.6367         |
| Multilingual   | Precision | 0.6783               | 0.6364         |
| Multilingual   | Recall    | 0.7319               | 0.7830         |
| Multilingual   | Accuracy  | 0.7331               | 0.6414         |
| English (ENGL) | F1 Score  | 0.5985               | 0.8144         |
| English (ENGL) | Precision | 0.9118               | 0.7399         |
| English (ENGL) | Recall    | 0.8857               | 0.9057         |
| English (ENGL) | Accuracy  | 0.9000               | 0.7937         |
| Finnish (FINN) | F1 Score  | 0.6146               | 0.6914         |
| Finnish (FINN) | Precision | 0.5544               | 0.8359         |
| Finnish (FINN) | Recall    | 0.4799               | 0.5895         |
| Finnish (FINN) | Accuracy  | 0.5070               | 0.5689         |
| German (GERM)  | F1 Score  | 0.4212               | 0.3705         |
| German (GERM)  | Precision | 0.3430               | 0.2736         |
| German (GERM)  | Recall    | 0.5313               | 0.6105         |
| German (GERM)  | Accuracy  | 0.6387               | 0.4987         |

