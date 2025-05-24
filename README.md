# Fake News Classifier - NLP Project

This repository contains a full Machine Learning pipeline to classify news as **real (1)** or **fake (0)** using Natural Language Processing (NLP) techniques, built entirely in Python.

The project includes preprocessing, feature engineering, model training with cross-validation, and prediction on unseen validation data.

---

## Project Structure

├── dataset/
│ ├── data.csv # Labeled dataset for training and validation
│ └── validation_data.csv # Unlabeled dataset to predict
├── fake_news_script.py # Main Python script
├── predictions_validation.csv # Final predictions on validation data
├── random_forest_model.pkl # Saved trained model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── README.md # Project description
├── requirements.txt
└── fake_news_nlp.ipynb # Jupyter Notebook step by step how-to

## How to Run

1. **Install the dependencies:**

```bash
pip install -r requirements.txt
python fake_news_script.py 
```

The script will:

    - Preprocess the data (tokenize, remove stopwords, apply stemming)
    - Combine title + text into a single feature
    - Apply TF-IDF vectorization + One-Hot Encoding for subject
    - Train a RandomForestClassifier
    - Perform 5-fold cross-validation with visual output
    - Save the model and vectorizer
    - Predict on validation_data.csv and save results in predictions_validation.csv
    - Evaluation: Cross-Validation

## Results: 
We use 5-fold cross-validation to evaluate the model’s ability to generalize.

    - Mean accuracy: e.g. 99.8%
    - Standard deviation: e.g. ± 0.0012
---

### Developed as part of the [Ironhack] Data Science program — NLP Challenge.
Authors: Affiong Akpanisong & Mo Benet 
