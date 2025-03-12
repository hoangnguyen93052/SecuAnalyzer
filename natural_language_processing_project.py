import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        text = text.lower()  # Lowercase text
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and non-alphabetic characters
        return text

    def tokenize(self, text):
        tokens = word_tokenize(text)  # Tokenize text
        filtered_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]  # Remove stopwords and stem
        return filtered_tokens

class SentimentAnalyzer:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        self.model = MultinomialNB()

    def preprocess_data(self):
        self.data['cleaned_text'] = self.data['text'].apply(lambda x: TextPreprocessor().clean_text(x))
        self.data['tokenized_text'] = self.data['cleaned_text'].apply(lambda x: TextPreprocessor().tokenize(x))
        X = self.vectorizer.fit_transform(self.data['tokenized_text'])
        y = self.data['label']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        self.plot_confusion_matrix(y_test, y_pred)

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

def main():
    # Load dataset
    url = 'https://some-dataset-url.com/data.csv'  # Example URL, replace with an actual dataset
    data = pd.read_csv(url)
    
    # Ensure that there are 'text' and 'label' columns
    assert 'text' in data.columns and 'label' in data.columns, "Data must contain 'text' and 'label' columns."

    # Initializing Sentiment Analyzer
    analyzer = SentimentAnalyzer(data)

    # Preprocess data
    X_train, X_test, y_train, y_test = analyzer.preprocess_data()

    # Train model
    analyzer.train_model(X_train, y_train)

    # Evaluate model
    analyzer.evaluate_model(X_test, y_test)

if __name__ == '__main__':
    main()