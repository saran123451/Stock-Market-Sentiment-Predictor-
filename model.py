from sklearn.base import is_regressor
import torch

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv('stock_news.csv.zip')

data = data.dropna()

# Tokenization and stopword removal
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation and numbers
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

data['headline'] = data['headline'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['headline'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer, TfidfTransformer, and LogisticRegression
model = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

new_headlines = [
    "Company X announces record profits for the quarter",
    "Economic downturn expected to hit the stock market"
]

# Preprocess the headlines
new_headlines = [preprocess_text(headline) for headline in new_headlines]

# Predict the sentiment
predictions = model.predict(new_headlines)

# Print the results
for headline, prediction in zip(new_headlines, predictions):
    sentiment = "Good" if prediction == 1 else "Bad"
    print(f'Headline: "{headline}" - Sentiment: {sentiment}')

pickle.dump(is_regressor, open('model.pkl', 'wb'))

# Load the model
model = pickle.load(open('model.pkl', 'rb'))