import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
    
    def load_and_prepare_data(self, file_path):
        """
        Load and prepare the IMDB dataset
        """
        # Read the CSV file
        df = pd.read_csv(file_path)
                
        # Print dataset information
        print("\nDataset Overview:")
        print(f"Total reviews: {len(df)}")
        print(f"Positive reviews: {len(df[df['sentiment'] == 1])}")
        print(f"Negative reviews: {len(df[df['sentiment'] == 0])}")
        
        return df
        
    def preprocess_text(self, X_train, X_test):
        """
        TEXT PREPROCESSING
        """
        print("Step 1: Preprocessing text data...")
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        return X_train_vectorized, X_test_vectorized
    
    def build_model(self, X_train_vectorized, y_train):
        """
        MODEL BUILDING
        """
        print("\nStep 2: Building and training the model...")
        
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        )
        self.model.fit(X_train_vectorized, y_train)
    
    def evaluate_model(self, X_test_vectorized, y_test):
        """
        MODEL EVALUATION
        """
        print("\nStep 3: Evaluating model performance...")
        
        y_pred = self.model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.2f}")
        
        print("\nDetailed Performance Metrics:")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def predict_new_review(self, review_text):
        """
        Make predictions on new reviews
        """
        review_vectorized = self.vectorizer.transform([review_text])
        prediction = self.model.predict(review_vectorized)[0]
        probability = self.model.predict_proba(review_vectorized)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[prediction]
        
        return sentiment, confidence

def main():
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load and prepare data
    print("Loading data...")
    df = analyzer.load_and_prepare_data('IMDB_dataset.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'],
        df['sentiment'],
        test_size=0.2,
        random_state=43
    )
    
    # Preprocess text
    X_train_vectorized, X_test_vectorized = analyzer.preprocess_text(X_train, X_test)
    
    # Build model
    analyzer.build_model(X_train_vectorized, y_train)
    
    # Evaluate model
    analyzer.evaluate_model(X_test_vectorized, y_test)
    
    # Test with new reviews
    print("\nTesting with new reviews:")
    test_reviews = [
        "This movie exceeded all my expectations. Absolutely brilliant!",
        "I regret watching this. Complete disappointment.",
        "The movie was not that bad",
        "I really dislike the movie story",
        "It was great"
    ]
    
    for review in test_reviews:
        sentiment, confidence = analyzer.predict_new_review(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence}%")

if __name__ == "__main__":
    main()