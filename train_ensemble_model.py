"""
Balanced training model with focus on real-world generalization
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
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

def load_and_preprocess_data(fake_path="../../Data/Fake.csv", true_path="../../Data/True.csv", max_samples_per_class=3000):
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
    
    # Remove rows with empty text after cleaning
    df = df[df['cleaned_text'].str.strip().str.len() > 10]
    print(f"Shape after removing empty texts: {df.shape}")
    
    # Add some synthetic examples of real news that might be confused with fake news
    print("Adding balanced examples...")
    
    # Real news that might sound sensational
    real_news_examples = [
        "Scientists make breakthrough discovery in cancer treatment that could save millions of lives.",
        "Earthquake in Pacific region measures 8.2 on Richter scale, tsunami warnings issued.",
        "Major technological company announces revolutionary new product that will change the industry.",
        "Stock market sees biggest single-day gain in history as economic indicators improve.",
        "President signs historic peace treaty ending decades-long conflict in troubled region."
    ]
    
    for example in real_news_examples:
        new_row = pd.DataFrame({
            'text': [example],
            'cleaned_text': [clean_text_conservative(example)],
            'label': [1]  # Real news
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Count class distribution
    class_counts = df['label'].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    return df

def train_ensemble_model():
    """Train an ensemble model for better generalization"""
    # Load and preprocess the data
    df = load_and_preprocess_data()
    
    # Split into features and target
    X = df['cleaned_text']
    y = df['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Create a more balanced TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,  # Use more features for better representation
        min_df=3,           # Ignore terms that appear in fewer than 3 documents
        max_df=0.8,         # Ignore terms that appear in more than 80% of documents
        ngram_range=(1, 2)  # Use both unigrams and bigrams for context
    )
    
    # TF-IDF transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"Number of features: {X_train_tfidf.shape[1]}")
    
    # Create individual models for the ensemble
    svm_model = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
    rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=20)
    
    # Create a voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft'  # Use probability estimates for voting
    )
    
    # Train the ensemble
    print("Training the ensemble model...")
    ensemble.fit(X_train_tfidf, y_train)
    
    # Evaluate the ensemble
    y_pred = ensemble.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Check false negative and false positive rates
    fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print(f"False Negative Rate: {fn_rate:.4f}")
    print(f"False Positive Rate: {fp_rate:.4f}")
    
    # Test on some real-world examples
    print("\nTesting on some real-world examples...")
    real_world_examples = [
        "The Federal Reserve announced a quarter-point interest rate increase today.",
        "Scientists have discovered a new species of frog in the Amazon rainforest.",
        "Breaking: Scientists discover that drinking coffee mixed with lemon juice can cure all types of cancer.",
        "Secret government documents reveal that aliens have been living among us for decades.",
        "The annual budget meeting is scheduled for next Tuesday at 2:00 PM."
    ]
    
    for i, example in enumerate(real_world_examples):
        # Preprocess text
        cleaned_text = clean_text_conservative(example)
        
        # Transform text
        text_vector = tfidf_vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = ensemble.predict(text_vector)[0]
        proba = ensemble.predict_proba(text_vector)[0]
        confidence = proba[int(prediction)]
        
        # Display results
        print(f"Example {i+1}:")
        print(f"Text: {example}")
        print(f"Predicted: {'REAL' if prediction == 1 else 'FAKE'}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 50)
    
    # Save the model and vectorizer
    print("\nSaving the ensemble model and vectorizer...")
    joblib.dump(ensemble, "svm_model.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    
    print("Model saved as 'svm_model.pkl' (ensemble model)")
    print("Vectorizer saved as 'tfidf_vectorizer.pkl'")
    
    return ensemble, tfidf_vectorizer

if __name__ == "__main__":
    print("Fake News Detection - Ensemble Model Training")
    print("=" * 50)
    model, vectorizer = train_ensemble_model()
    print("\nEnsemble training complete! Your model is now better at generalizing to real-world examples.")
    print("Run verify_model.py to see how it performs on test examples.")
