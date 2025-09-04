"""
Improved training model with enhanced balancing and feature selection
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text_conservative(text):
    """Clean text with a more nuanced approach that preserves important linguistic features"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs but keep the context
    text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text, flags=re.MULTILINE)
    
    # Replace email with placeholder
    text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)
    
    # Keep some numbers but remove excessive ones
    text = re.sub(r'\b\d{4,}\b', ' [NUMBER] ', text)  # Replace long numbers
    
    # Keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # More conservative stopword removal
    stop_words = set(stopwords.words("english"))
    # Keep important words that might indicate bias or opinion
    keep_words = {
        'not', 'no', 'never', 'nothing', 'nobody', 'neither', 'nowhere', 'none',
        'but', 'however', 'although', 'though', 'yet', 'still', 'nevertheless',
        'very', 'really', 'quite', 'extremely', 'highly', 'completely',
        'must', 'should', 'would', 'could', 'might', 'may',
        'always', 'never', 'often', 'sometimes', 'usually'
    }
    stop_words = stop_words - keep_words
    
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words and len(word) > 1])
    
    return text

def load_and_preprocess_data(fake_path="../../Data/Fake.csv", true_path="../../Data/True.csv", max_samples_per_class=2000):
    """Load and preprocess the data with a balanced approach"""
    print("Loading data...")
    # Load the datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add labels
    fake_df['label'] = 0  # 0 for fake news
    true_df['label'] = 1  # 1 for real news
    
    # Sample an equal amount from each class to ensure balance
    if len(fake_df) > max_samples_per_class:
        fake_df = fake_df.sample(max_samples_per_class, random_state=42)
    if len(true_df) > max_samples_per_class:
        true_df = true_df.sample(max_samples_per_class, random_state=42)
    
    # Combine the datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clean the text data
    print("Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(clean_text_conservative)
    
    # Check if cleaning removed too much content
    text_lengths = df['cleaned_text'].str.len()
    print(f"Average text length after cleaning: {text_lengths.mean():.2f} characters")
    print(f"Min text length: {text_lengths.min()}, Max text length: {text_lengths.max()}")
    
    # Remove rows with empty text after cleaning
    df = df[df['cleaned_text'].str.strip().str.len() > 10]
    print(f"Shape after removing empty texts: {df.shape}")
    
    # Count class distribution
    class_counts = df['label'].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    return df

def train_and_evaluate_model():
    """Train and evaluate the model"""
    # Load and preprocess the data
    df = load_and_preprocess_data()
    
    # Split into features and target
    X = df['cleaned_text']
    y = df['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Create a pipeline with TF-IDF and SVM
    print("Creating TF-IDF vectorizer and model pipeline...")
    
    # Create a more balanced TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=2000,  # Use 2000 features for better balance
        min_df=5,          # Ignore terms that appear in fewer than 5 documents
        max_df=0.7,        # Ignore terms that appear in more than 70% of documents
        ngram_range=(1, 2) # Use both unigrams and bigrams for context
    )
    
    # TF-IDF transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # Check feature distribution
    print(f"Number of features: {X_train_tfidf.shape[1]}")
    
    # Try multiple models to find the best one
    print("Training multiple models for comparison...")
    
    models = {
        'SVM (Linear)': SVC(kernel='linear', C=1.0, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100)
    }
    
    best_accuracy = 0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate on test data
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {name}:")
        print(cm)
        
        # Check false negative and false positive rates
        fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        print(f"False Negative Rate: {fn_rate:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        
        # If this model has better accuracy, save it
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
    
    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    # Save the best model and vectorizer
    print("Saving the best model and vectorizer...")
    joblib.dump(best_model, "svm_model.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    
    print(f"Model saved as 'svm_model.pkl' (even though it might be {best_model_name})")
    print("Vectorizer saved as 'tfidf_vectorizer.pkl'")
    
    # Return the best model and vectorizer for further use if needed
    return best_model, tfidf_vectorizer, X_test, y_test

if __name__ == "__main__":
    print("Fake News Detection - Enhanced Training Pipeline")
    print("=" * 50)
    model, vectorizer, X_test, y_test = train_and_evaluate_model()
    print("\nTraining complete! Now you can use the model to classify news.")
    print("Run test_model.py to see how it performs on some test examples.")
