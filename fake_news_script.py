import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

df = pd.read_csv('./dataset/data.csv', encoding="ISO-8859-1")
df['combined'] = df['title'] + " " + df['text']
y = df['label']
df.drop(columns=['label', 'title', 'text', 'date'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

X_train['clean_combined'] = X_train['combined'].apply(preprocess_text)
X_test['clean_combined'] = X_test['combined'].apply(preprocess_text)

subject_dummies_train = pd.get_dummies(X_train['subject'], prefix='subject', dtype=int)
subject_dummies_test = pd.get_dummies(X_test['subject'], prefix='subject', dtype=int)

# When doing pd.get_dummies on test it can happen that the X_Test doesn't have all the subjects
subject_dummies_test = subject_dummies_test.reindex(columns=subject_dummies_train.columns, fill_value=0)

X_train = pd.concat([X_train.drop(columns=['subject', 'combined']), subject_dummies_train], axis=1)
X_test = pd.concat([X_test.drop(columns=['subject', 'combined']), subject_dummies_test], axis=1)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_text = tfidf.fit_transform(X_train['clean_combined'])
X_test_text = tfidf.transform(X_test['clean_combined'])

X_train_other = X_train.drop(columns=['clean_combined'])
X_test_other = X_test.drop(columns=['clean_combined'])

# Combine TF-IDF vectors with one-hot features
X_train_final = hstack([X_train_text, X_train_other.values])
X_test_final = hstack([X_test_text, X_test_other.values])

model = RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=42)
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
