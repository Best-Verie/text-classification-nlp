# Cancer Text Classification Using Deep Learning and Machine Learning

Classifying biomedical research papers into three cancer types (Thyroid, Colon, Lung) using four different models — GRU, LSTM, SimpleRNN, and Logistic Regression — each evaluated with multiple word embedding techniques.

## Dataset

- **Source:** 7,570 cancer research documents from Kaggle
- **Classes:** Thyroid Cancer (2,810), Colon Cancer (2,580), Lung Cancer (2,180)
- **Split:** 70% train / 15% validation / 15% test (stratified)

## Text Preprocessing

All models share the same preprocessing pipeline:
- Lowercasing, URL/email removal, special character cleanup
- Stopword removal (NLTK English stopwords)
- Lemmatization (WordNet)
- Minimum token length filtering

## Shared Embedding: Skip-gram (Word2Vec)

A shared Skip-gram embedding was trained on the full corpus using Gensim (`vector_size=100`, `window=5`, `min_count=5`, `sg=1`) and saved to `preprocesseddata/` for all team members to load into their models.

## Models and Results

### GRU (Gated Recurrent Unit)

Stacked 2-layer GRU (128 → 64 hidden units) with dropout 0.3, built in PyTorch. Trained with Adam optimizer and early stopping (patience=3).

| Embedding | Accuracy | F1-Score |
|-----------|----------|----------|
| TF-IDF | 96.92% | 0.9692 |
| Skip-gram | 96.65% | 0.9666 |
| CBOW | 98.59% | 0.9859 |
| **GloVe** | **98.77%** | **0.9877** |

### LSTM (Long Short-Term Memory)

Stacked 2-layer LSTM (128 → 64 hidden units) with dropout 0.3, built in TensorFlow/Keras. Trained with Adam optimizer and ReduceLROnPlateau.

| Embedding | Accuracy | F1-Score |
|-----------|----------|----------|
| **TF-IDF** | **97.42%** | **0.9742** |
| Skip-gram | 97.16% | 0.9716 |
| CBOW | 97.36% | 0.9736 |

### SimpleRNN (Vanilla RNN)

Single-layer SimpleRNN (64 units) with dropout 0.3, built in TensorFlow/Keras. Trained with Adam optimizer for 30 epochs.

| Embedding | Accuracy | F1-Score |
|-----------|----------|----------|
| **TF-IDF** | **85.46%** | **0.8542** |
| Skip-gram | 75.53% | 0.7555 |
| CBOW | 70.57% | 0.7007 |

### Logistic Regression

Multinomial Logistic Regression with GridSearchCV over regularization parameter C, using scikit-learn (solver=saga).

| Embedding | Accuracy | F1-Score |
|-----------|----------|----------|
| **TF-IDF** | **97.09%** | **0.97** |
| Skip-gram (W2V) | 81.11% | 0.81 |
| CBOW (W2V) | 79.61% | 0.80 |

## Overall Comparison

| Model | Best Embedding | Accuracy | F1-Score |
|-------|---------------|----------|----------|
| **GRU** | **GloVe** | **98.77%** | **0.9877** |
| LSTM | TF-IDF | 97.42% | 0.9742 |
| Logistic Regression | TF-IDF | 97.09% | 0.97 |
| SimpleRNN | TF-IDF | 85.46% | 0.8542 |

**Best overall model: GRU + GloVe** (F1 = 0.9877). The GRU's gating mechanism combined with GloVe's pre-trained semantic representations achieved the highest performance. Lung Cancer was the easiest class to classify across all models due to its highly distinctive vocabulary (e.g., "nsclc", "nonsmallcell"). TF-IDF consistently performed well as a baseline embedding, especially for simpler models like Logistic Regression and SimpleRNN.

## Project Structure

```
scripts/
  GRU_Cancer_Classification.ipynb        # GRU model training and evaluation
  LSTM_Cancer_Classification.ipynb       # LSTM model training and evaluation
  text-classification-rnn.ipynb          # SimpleRNN model training and evaluation
  logistic_regression_Classification.ipynb  # Logistic Regression experiments
  skipgram_preprocessing.ipynb           # Shared Skip-gram embedding generation
  data_preprocessing.ipynb               # Initial data exploration and EDA
preprocesseddata/                        # Shared embedding artifacts for all team members
```
