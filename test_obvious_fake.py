import joblib
import pandas as pd
import re
from nltk.corpus import stopwords

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

def test_obvious_fake_news():
    """Test with obviously fake news examples"""
    print("🚨 Testing with OBVIOUSLY fake news examples...")
    
    # Load models
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    obvious_fake_examples = [
        "BREAKING: Aliens land in New York City, demand to speak to the president immediately! Government tries to cover up the truth!",
        "SHOCKING: Doctors discover that drinking bleach cures all diseases! Big pharma doesn't want you to know this one simple trick!",
        "UNBELIEVABLE: Local man grows 10 feet tall after eating this magical fruit! Scientists are baffled by this incredible transformation!",
        "EXPOSED: The earth is actually flat and NASA has been lying to us for decades! Here's the proof they don't want you to see!",
        "MIRACLE: Woman gives birth to a baby dinosaur! Evolution scientists are completely stumped by this amazing discovery!",
        "INCREDIBLE: Time traveler from 2050 warns us about the coming zombie apocalypse! Government officials refuse to comment!",
        "AMAZING: Scientists prove that chocolate cake is actually healthier than vegetables! Nutritionists everywhere are shocked!",
        "BREAKING: Moon is made of cheese confirmed by new NASA mission! Astronauts bring back samples for testing!"
    ]
    
    print("\n" + "="*80)
    
    real_predictions = 0
    fake_predictions = 0
    
    for i, text in enumerate(obvious_fake_examples, 1):
        # Clean the text
        cleaned_text = clean_text_conservative(text)
        print(f"\nExample {i}:")
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned_text}")
        
        # Check if text becomes empty after cleaning
        if not cleaned_text.strip():
            print("❌ Text became empty after cleaning!")
            continue
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        
        # Get confidence
        if hasattr(model, 'decision_function'):
            confidence = model.decision_function(text_vector)[0]
            confidence_abs = abs(confidence)
        else:
            confidence = "N/A"
            confidence_abs = "N/A"
        
        result = "REAL" if prediction == 1 else "FAKE"
        
        if prediction == 1:
            real_predictions += 1
        else:
            fake_predictions += 1
        
        print(f"Prediction: {result} | Confidence: {confidence} | Abs Confidence: {confidence_abs}")
        
        if prediction == 1:  # If predicted as real
            print("🚨 WARNING: Obviously fake news predicted as REAL!")
        else:
            print("✅ Correctly identified as fake")
    
    print("\n" + "="*80)
    print(f"📊 Summary of {len(obvious_fake_examples)} obviously fake examples:")
    print(f"   Predicted as REAL: {real_predictions}")
    print(f"   Predicted as FAKE: {fake_predictions}")
    
    if real_predictions > fake_predictions:
        print("🚨 PROBLEM: Model is heavily biased towards predicting REAL news!")
        print("   Even obviously fake news is being classified as real.")
    elif real_predictions > 0:
        print("⚠️  CONCERN: Some obviously fake news is being classified as real.")
    else:
        print("✅ Good: All obviously fake news was correctly identified.")

if __name__ == "__main__":
    test_obvious_fake_news()
