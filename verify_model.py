"""
Verify Model - Test the model on real-world examples with consistent preprocessing
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from nltk.corpus import stopwords
import nltk

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text_conservative(text):
    """Matches the exact preprocessing from train_model.py"""
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

def verify_model():
    """Test the model with real news examples"""
    print("Loading model and vectorizer...")
    try:
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return
    
    # Test examples - include both real and fake news examples
    test_examples = [
        # Real news examples
        {
            "text": "The Federal Reserve announced a quarter-point interest rate increase today, bringing the target range to 1.75%-2.00%. This decision was based on strong economic growth and low unemployment rates.",
            "label": 1
        },
        {
            "text": "Scientists have discovered a new species of frog in the Amazon rainforest. The discovery highlights the rich biodiversity of the region and the importance of conservation efforts.",
            "label": 1
        },
        # Fake news examples
        {
            "text": "Breaking: Scientists discover that drinking coffee mixed with lemon juice can cure all types of cancer overnight. The pharmaceutical industry has been hiding this miracle cure for decades.",
            "label": 0
        },
        {
            "text": "Secret government documents reveal that aliens have been living among us for decades and have infiltrated the highest levels of government. The president is reportedly in direct communication with their leader.",
            "label": 0
        },
        # Neutral factual text
        {
            "text": "The annual budget meeting is scheduled for next Tuesday at 2:00 PM in Conference Room A. All department heads are required to attend and bring their quarterly reports.",
            "label": 1
        }
    ]
    
    # Process and test each example
    predictions = []
    actual_labels = []
    
    print("\nTesting model on examples...\n")
    print("-" * 70)
    
    for i, example in enumerate(test_examples, 1):
        # Preprocess text
        cleaned_text = clean_text_conservative(example["text"])
        
        # Transform text
        text_vector = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        
        # Different models have different ways to get confidence scores
        if hasattr(model, 'decision_function'):
            # For SVM
            confidence = abs(model.decision_function(text_vector)[0])
        elif hasattr(model, 'predict_proba'):
            # For Random Forest, Logistic Regression, etc.
            proba = model.predict_proba(text_vector)[0]
            # Use the probability of the predicted class
            confidence = proba[int(prediction)]
        else:
            confidence = 0.5  # Default if no confidence method available
        
        # Store results
        predictions.append(prediction)
        actual_labels.append(example["label"])
        
        # Display results
        print(f"Example {i}:")
        print(f"Text: {example['text'][:100]}...")
        print(f"Actual: {'REAL' if example['label'] == 1 else 'FAKE'}")
        print(f"Predicted: {'REAL' if prediction == 1 else 'FAKE'}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'✓' if prediction == example['label'] else '✗'}")
        print("-" * 70)
    
    # Calculate overall metrics
    accuracy = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions, target_names=['FAKE', 'REAL'])
    conf_matrix = confusion_matrix(actual_labels, predictions)
    
    # Print results
    print("\nOverall Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nConfusion Matrix Explanation:")
    print("True Negatives (Correctly identified FAKE news):", conf_matrix[0, 0])
    print("False Positives (FAKE news classified as REAL):", conf_matrix[0, 1])
    print("False Negatives (REAL news classified as FAKE):", conf_matrix[1, 0])
    print("True Positives (Correctly identified REAL news):", conf_matrix[1, 1])

if __name__ == "__main__":
    verify_model()
