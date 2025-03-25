﻿# Multilingual-Toxicity-classification-transformer-model
Toxic Text Classification using Transformer
This project implements a Transformer-based model to classify text as toxic or non-toxic. It features a custom Byte Pair Encoding (BPE) tokenizer and a Transformer encoder architecture, built using PyTorch. The model is trained on a dataset of labeled text and can be used to predict toxicity in new text samples.

Features
Custom BPE tokenizer for text preprocessing
Transformer encoder with multi-head self-attention
Class imbalance handling with weighted loss
Efficient data loading with multi-worker DataLoader
Model training and evaluation with accuracy and F1 score metrics
Predictions saved in TSV format
Requirements
Python 3.6+
PyTorch 1.7+
tokenizers
pandas
numpy
matplotlib
scikit-learn
tqdm
Setup
Follow these steps to set up the project environment:

Clone the Repository:
bash

Collapse

Wrap

Copy
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Create a Virtual Environment:
bash

Collapse

Wrap

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Note: Create a requirements.txt file with the listed dependencies if it doesn’t exist:
text

Collapse

Wrap

Copy
torch>=1.7.0
tokenizers
pandas
numpy
matplotlib
scikit-learn
tqdm
Dataset
The project expects datasets in TSV format with two columns:

text: The text to classify
label: The label (0 for non-toxic, 1 for toxic)
Prepare the following files:

train.tsv: Training data
dev.tsv: Development/validation data
test.tsv: Test data for predictions
Place these files in the project root directory.

Training the Tokenizer
The project uses a custom BPE tokenizer trained on your text data:

Prepare Tokenizer Corpus: Create a file named tokenizer_corpus.txt with text data for training the tokenizer (e.g., a large collection of text samples).
Train the Tokenizer: Run the following Python code (either in a script or a Jupyter notebook cell):
python

Collapse

Wrap

Copy
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(["tokenizer_corpus.txt"], trainer)
tokenizer.save("bpe_tokenizer.json")
This generates bpe_tokenizer.json, which will be used by the model.
Running the Code
The project is implemented in a Jupyter notebook (toxic_classification.ipynb). To run it:

Open the Notebook:
bash

Collapse

Wrap

Copy
jupyter notebook toxic_classification.ipynb
Run the Cells: Execute the cells in order:
Install dependencies (if not already installed)
Train the tokenizer (if bpe_tokenizer.json is not available)
Load and preprocess the datasets
Train the model (saves the best model to best_transformer_model.pth)
Evaluate and make predictions on the test set (saves to predictions.tsv)
Alternative: Convert to Script: Convert the notebook to a Python script and run it:
bash

Collapse

Wrap

Copy
jupyter nbconvert --to script toxic_classification.ipynb
python toxic_classification.py
Note: You may need to adjust the script for interactive parts or errors (e.g., the model loading issue in the notebook).
Model Architecture
The Transformer encoder has the following configuration:

Vocabulary Size: 30,000
Embedding Size: 128
Number of Layers: 3
Number of Attention Heads: 4
Forward Expansion: 2
Dropout: 0.3
Maximum Sequence Length: 128
The model uses the [CLS] token’s output for classification into two classes (toxic or non-toxic).

Training
The training process:

Runs for 5 epochs
Uses Adam optimizer with a learning rate of 1e-4
Applies a weighted CrossEntropyLoss (weights: [1.0, 1.7] for classes 0 and 1, respectively) to handle class imbalance
Saves the best model based on validation loss to best_transformer_model.pth
Making Predictions
After training, the model generates predictions on the test set:

Loads the best model from best_transformer_model.pth
Processes test.tsv
Saves predictions to predictions.tsv with a predicted column (0 or 1)
Note: The notebook has a RuntimeError when loading the model due to mismatched architecture parameters. Ensure the saved model matches the defined TransformerEncoder architecture (e.g., embed_size=128).

Notes
Class Imbalance: The model uses class weights [1.0, 1.7] to address imbalance (adjust based on your dataset’s ratio).
DataLoader Optimization: Configured with 8 workers, pinned memory, and prefetching for faster data loading. Adjust NUM_WORKERS based on your CPU cores.
Error Handling: If you encounter model loading errors, verify that the saved model’s architecture matches the code (e.g., embed_size, num_layers).
Results
The model evaluates performance using:

Accuracy
Weighted F1 score
Add your results to the README after training (e.g., "Accuracy: 0.85, F1 Score: 0.82").

License
This project is licensed under the MIT License.

This README provides a complete guide to set up and use your project. Replace yourusername/your-repo-name with your actual GitHub repository URL, and update the "Results" section with your model’s performance metrics once available. If you resolve the model loading error in the notebook, you can simplify the prediction instructions accordingly.
