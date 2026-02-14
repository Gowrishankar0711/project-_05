#  Comment Toxicity Detection Project

##  Project Overview

The **Comment Toxicity Detection System** is a Deep Learning-based web application that detects whether a user comment is **Toxic** or **Non-Toxic**.

This project uses **Natural Language Processing (NLP)** and Deep Learning models (LSTM, CNN, BiLSTM) to classify text comments. The application is deployed using **Streamlit** for real-time predictions.

---

## Dataset

The project uses two datasets:
- **train.csv**: Training dataset containing labeled comments
- **test.csv**: Test dataset for evaluation

### Dataset Statistics
- Training samples: ~159,571 comments
- Multiple binary labels for different toxicity types:
  - **Toxic**: 15,294 positive samples (~9.6%)
  - **Severe Toxic**: 1,595 positive samples (~1%)
  - **Obscene**: Imbalanced distribution
  - **Threat**: Minimal samples
  - **Insult**: Moderate distribution
  - **Identity Hate**: Low frequency

## Dependencies

```
pandas
numpy
matplotlib
nltk
scikit-learn
tensorflow
keras
seaborn
missingno
```

## Data Preprocessing Pipeline

### 1. **Text Cleaning**
   - Convert text to lowercase
   - Remove special characters and punctuation using regex
   - Retain only alphabetic characters and spaces

### 2. **Tokenization & Stopword Removal**
   - Tokenize cleaned text using NLTK's `word_tokenize()`
   - Remove English stopwords using NLTK stopwords corpus
   - Result: meaningful tokens only

### 3. **Vectorization**
   - Use Keras `Tokenizer` with 10,000 most frequent words
   - Convert text sequences to numerical sequences
   - Pad all sequences to fixed length of 100 tokens

### 4. **Train-Test Split**
   - 80% training data
   - 20% testing data
   - Random state: 42 (reproducible results)

## Models Implemented

### 1. **LSTM (Long Short-Term Memory)**

**Architecture:**
```
- Embedding Layer: (vocab_size=10000, embedding_dim=128)
- LSTM Layer: 64 units
- Dropout: 0.5 (regularization)
- Dense Output: 1 unit with sigmoid activation
```

**Compilation:**
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

**Training Configuration:**
- Epochs: 5
- Batch Size: 32
- Validation: 20% of training data

---

### 2. **CNN (Convolutional Neural Network)**

**Architecture:**
```
- Embedding Layer: (vocab_size=10000, embedding_dim=128)
- Conv1D Layer: 64 filters, kernel size=5, ReLU activation
- GlobalMaxPooling1D: Reduces feature maps
- Dense Output: 1 unit with sigmoid activation
```

**Compilation:**
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

**Training Configuration:**
- Epochs: 5
- Batch Size: 32
- Validation: 20% of training data

---

### 3. **BiLSTM (Bidirectional LSTM)** - Primary Model ⭐

**Architecture:**
```
- Embedding Layer: (vocab_size=10000, embedding_dim=128)
- Bidirectional LSTM: 64 units, processes sequence forward and backward
- Dropout: 0.5
- Dense Layer: 64 units with ReLU activation
- Dropout: 0.3
- Dense Output: 1 unit with sigmoid activation
```

**Compilation:**
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

**Training Configuration:**
- Epochs: 5
- Batch Size: 32
- Validation: 20% of training data

**Threshold Tuning:**
- Default threshold: 0.5
- Adjusted threshold: 0.35 for improved recall

## Model Evaluation

### Metrics Used

1. **Classification Report**
   - Precision: True positive rate among predicted positives
   - Recall: True positive rate among actual positives
   - F1-Score: Harmonic mean of precision and recall
   - Support: Number of test samples per class

2. **Confusion Matrix**
   - True Positives (TP)
   - True Negatives (TN)
   - False Positives (FP)
   - False Negatives (FN)

3. **ROC-AUC Curve**
   - Area Under the ROC Curve metric
   - Plots False Positive Rate vs True Positive Rate
   - Evaluates classifier performance across thresholds

### Visualizations
- Confusion matrix heatmaps using Seaborn
- ROC curves for model comparison
- Classification reports for detailed metrics

## Model Artifacts

All trained models and preprocessing utilities are saved for inference:

```
toxicity_model.keras           # LSTM model
toxicity_model_BiLSTM.keras    # BiLSTM model (recommended)
tokenizer.pkl                  # Fitted tokenizer for text vectorization
```

## Key Findings

### Model Performance
- **BiLSTM** provides the best balance between precision and recall
- Bidirectional processing captures both forward and backward context
- Threshold adjustment (0.35) improves recall on imbalanced classes
- ROC-AUC score indicates strong discriminative ability

### Data Insights
- Significant class imbalance in toxicity labels
- Most comments are non-toxic (benign class dominates)
- Rare categories (threat, severe_toxic) pose classification challenges


   - Increase training epochs (currently 5)
   - Experiment with different embedding dimensions
   - Use pre-trained embeddings (GloVe, Word2Vec)
   - Ensemble multiple models
 
## streamlit

The app.py is a Streamlit web application that provides an interactive dashboard for comment toxicity detection. Here are the key features

### Main Components:
- Model & Tokenizer Loading: Loads the pre-trained BiLSTM model and tokenizer with caching
- Text Preprocessing: Implements cleaning, tokenization, stopword removal, and vectorization
- Dashboard with 4 Tabs:
- Predict Tab: Single comment analysis with toxicity score, classification (Toxic/Non-Toxic), confidence percentage, and text preprocessing visualization
- Insights Tab: Data statistics, class distribution charts, model architecture details, and performance metrics
- Bulk Upload Tab: CSV file upload for batch predictions with results table, visualizations, and CSV download option
- About Tab: Complete documentation on features, technical details, usage guide, and model improvements

### Key Features:
- Real-time Predictions: Analyzes comments instantly with toxicity scores and confidence levels
- Visualizations: Pie charts, bar charts showing class distribution and analysis results
- Batch Processing: Upload CSV with comments for bulk analysis with progress tracking
- Text Analysis: Shows original → cleaned → processed text transformation steps
- Sortable Results: Results can be sorted by score, prediction, or index
- Export Capability: Download predictions as CSV file
- Error Handling: Graceful handling of missing files and data validation

The app is production-ready with custom CSS styling, responsive layouts, caching for performance, and comprehensive error messages

