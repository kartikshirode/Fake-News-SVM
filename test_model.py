"""
Test script for the advanced fake news detector
"""
import joblib
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def test_model():
    """Test the model with various real-world examples"""
    # Load models
    try:
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
    except FileNotFoundError:
        print("❌ Model files not found. Please run train_model.py first!")
        return
    
    # Test cases - mix of real and potentially fake news
    test_cases = [
        {
            "text": "Scientists at MIT have developed a new quantum computer that can solve complex mathematical problems in seconds. The breakthrough could revolutionize cryptography and drug discovery research.",
            "expected": "Should be classified as REAL news"
        },
        {
            "text": "BREAKING: Aliens have landed in Central Park and are demanding to speak with world leaders immediately. Government officials are trying to cover up the story.",
            "expected": "Should be classified as FAKE news"
        },
        {
            "text": "The Federal Reserve announced today that they will be adjusting interest rates in response to recent inflation data. This decision comes after careful analysis of economic indicators.",
            "expected": "Should be classified as REAL news"
        },
        {
            "text": "Local man discovers ancient treasure worth millions in his backyard while planting tomatoes. Archaeologists are baffled by the incredible find.",
            "expected": "Should be classified as FAKE news"
        },
        {
            "text": "New study published in Nature journal shows promising results for a potential Alzheimer's treatment. Clinical trials will begin next year across multiple research centers.",
            "expected": "Should be classified as REAL news"
        }
    ]
    
    print("🔍 Testing our advanced fake news detector!\n")
    print("=" * 60)
    
    all_predictions = []
    all_expecteds = []
    
    for i, case in enumerate(test_cases, 1):
        # Preprocess the text - using the SAME function as in training
        processed_text = clean_text_conservative(case["text"])
        
        # Make prediction
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)[0]
        
        # Get probability/confidence if available
        confidence = 0
        if hasattr(model, "decision_function"):
            confidence = abs(model.decision_function(text_vector)[0])
        elif hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vector)[0]
            confidence = probs[1] if prediction == 1 else probs[0]
            confidence = confidence * 2  # Scale to similar range as decision_function
        
        # Format results
        result = "✅ REAL NEWS" if prediction == 1 else "❌ FAKE NEWS"
        expected = 1 if "REAL" in case["expected"] else 0
        
        all_predictions.append(prediction)
        all_expecteds.append(expected)
        
        print(f"Test Case {i}:")
        print(f"📰 Text: {case['text'][:80]}...")
        print(f"🎯 Expected: {case['expected']}")
        print(f"🤖 AI Prediction: {result}")
        print(f"📊 Confidence: {confidence:.2f}")
        print("-" * 60)
    
    # Overall accuracy
    correct = sum(1 for p, e in zip(all_predictions, all_expecteds) if p == e)
    total = len(all_predictions)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 Overall Test Accuracy: {accuracy:.2f} ({correct}/{total} correct)")
    
    # Create confusion matrix if we have at least 2 examples
    if len(all_predictions) >= 2:
        labels = ["Fake News", "Real News"]
        cm = confusion_matrix(all_expecteds, all_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Test Cases Confusion Matrix')
        plt.ylabel('Expected')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('test_results.png')
        print("📈 Confusion matrix saved as 'test_results.png'")

if __name__ == "__main__":
    test_model()
