import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def download_imdb_data():
    """
    Download a small subset of IMDB reviews for training
    """
    # Create a simple dataset
    reviews = []
    sentiments = []
    
    # Positive reviews
    positive_reviews = [
        "This movie was absolutely amazing! The acting was superb.",
        "One of the best films I've seen this year. Highly recommended!",
        "Great storyline and wonderful cinematography.",
        "A masterpiece that keeps you engaged throughout.",
        "Brilliant performance by the entire cast."
    ]
    
    # Negative reviews
    negative_reviews = [
        "Complete waste of time. The plot made no sense.",
        "Terrible acting and poor direction. Would not recommend.",
        "I was really disappointed with this movie.",
        "The worst film I've seen in years.",
        "Save your money and skip this one."
    ]
    
    reviews.extend(positive_reviews)
    reviews.extend(negative_reviews)
    sentiments.extend([1] * len(positive_reviews))  # 1 for positive
    sentiments.extend([0] * len(negative_reviews))  # 0 for negative
    
    # Create a DataFrame
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def prepare_data():
    # Download and load the data
    print("Downloading movie reviews dataset...")
    df = download_imdb_data()
    
    # Display some basic information about the dataset
    print("\nDataset Overview:")
    print(f"Total number of reviews: {len(df)}")
    print(f"Number of positive reviews: {len(df[df['sentiment'] == 1])}")
    print(f"Number of negative reviews: {len(df[df['sentiment'] == 0])}")
    
    # Display a few examples
    print("\nExample Reviews:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = prepare_data()
    
    # Save the data
    df.to_csv('movie_reviews.csv', index=False)
    print("\nData saved to 'movie_reviews.csv'")