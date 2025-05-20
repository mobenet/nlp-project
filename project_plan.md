# PROJECT PLAN | NLP Fake News Classifier


## Phase 1: Data Loading and Exploration
    1. Import necessary libraries
    2. Load/Read dataset/data.csv
    3. Initial data inspection:
        - Check null values
        - Check class balance (label: 0 = fake, 1 = real)
        - View text examples
    4. Plan with pseudocode: Define a rough sketch of steps before implementing code

## Phase 2: Text Preprocessing

    1. Preprocessing 
        - Convert to lowercase
        - Clean text with Regex: remove special characters, numbers, links
        - Remove stopwords using nltk
        - Apply stemming or lemmatization (SnowballStemmer or WordNetLemmatizer)
                - stemming: faster but can distort words
                - lemmatization: slower, but gives valid words (chosen)

    

## Phase 3: Feature Engineering
    Convert preprocessed text to vectors using:
        - TF-IDF (TfidfVectorizer)
        - Optional: Word2Vec embeddings (if using neural networks or contextual models)
    For categorical features like subject, apply:
        - One-hot encoding or label encoding

    1. Select which columns to use (title, text, etc.)
    2. Split data.csv into training and test sets (train_test_split)
    3. Vectorize the text data
    4. Encode the labels if needed (LabelEncoder)

## Phase 4: Model Training

    1. Try several classifiers:
        - Logistic Regression
        - Naive Bayes
        - Random Forest (choosen)

    2. Evaluation:
        - Accuracy
        - Classification report
        - Confusion matrix

## Phase 5: Validation and Prediction

    1. Load validation_data.csv
    2. Apply the same preprocessing
    3. Use your model to make predictions
    4. Replace label == 2 with 0 or 1 using model predictions
    5. Save the CSV with the same format and column separator as the original

## Phase 6: Presentation and Submission

    1. Well-documented Python code
    2. CSV file with predictions
    3. Brief report with your estimated accuracy
    4. 10-minute presentation explaining your pipeline and results