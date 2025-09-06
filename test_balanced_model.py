import joblib
import pandas as pd
import re
from nltk.corpus import stopwords

def clean_text_conservative(text):
    """Less aggressive text cleaning to preserve fake news indicators"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs but keep the context
    text = re.sub(r'http\S+|www\S+|https\S+', ' url ', text, flags=re.MULTILINE)
    
    # Replace email with placeholder
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # Keep most numbers - they might be important
    text = re.sub(r'\b\d{8,}\b', ' longnumber ', text)  # Only replace very long numbers
    
    # Keep more punctuation - exclamation marks are important for fake news detection!
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:\-\'\"]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Much more conservative stopword removal - keep bias indicators
    stop_words = set(stopwords.words("english"))
    # Keep important words that indicate sensationalism or bias
    keep_words = {
        'not', 'no', 'never', 'nothing', 'nobody', 'neither', 'nowhere', 'none',
        'but', 'however', 'although', 'though', 'yet', 'still', 'nevertheless',
        'very', 'really', 'quite', 'extremely', 'highly', 'completely', 'totally',
        'must', 'should', 'would', 'could', 'might', 'may', 'will', 'shall',
        'always', 'never', 'often', 'sometimes', 'usually', 'definitely',
        'shocking', 'amazing', 'incredible', 'unbelievable', 'breaking', 'exposed',
        'secret', 'hidden', 'truth', 'revealed', 'discover', 'scientists', 'doctors',
        'government', 'officials', 'experts', 'study', 'research'
    }
    stop_words = stop_words - keep_words
    
    words = text.split()
    # Keep words that are at least 2 characters (more lenient)
    text = ' '.join([word for word in words if word not in stop_words and len(word) > 1])
    
    return text

def test_balanced_model():
    """Test the new balanced model"""
    print("🔍 Testing the NEW BALANCED model...")
    
    # Load balanced models
    try:
        model = joblib.load("svm_model_balanced.pkl")
        vectorizer = joblib.load("tfidf_vectorizer_balanced.pkl")
        print("✅ Balanced models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading balanced models: {e}")
        return
    
    # Test with obvious fake news examples
    obvious_fake_examples = [
        "BREAKING: Aliens land in New York City, demand to speak to the president immediately!",
        "SHOCKING: Doctors discover that drinking bleach cures all diseases!",
        "UNBELIEVABLE: Local man grows 10 feet tall after eating magical fruit!",
        "EXPOSED: The earth is actually flat and NASA has been lying to us!",
        "MIRACLE: Woman gives birth to a baby dinosaur!",
        "INCREDIBLE: Time traveler from 2050 warns about zombie apocalypse!",
        "AMAZING: Chocolate cake is healthier than vegetables, scientists confirm!",
        "BREAKING: Moon is made of cheese confirmed by NASA mission!"
    ]
    
    # Test with obvious real news examples
    obvious_real_examples = [
        "The Federal Reserve announced a 0.25% interest rate increase today.",
        "President Biden met with European leaders to discuss trade agreements.",
        "The stock market closed up 2% today amid positive economic indicators.",
        "Weather forecast shows heavy rain expected this weekend.",
        "Local government approves new infrastructure spending bill.",
        "University researchers publish study on climate change impacts.",
        "New hospital opens in downtown area to serve community.",
        "City council meeting scheduled for next Tuesday evening."
    ]
    
    print("\n🧪 Testing with OBVIOUSLY FAKE examples:")
    print("="*60)
    
    fake_correct = 0
    for i, text in enumerate(obvious_fake_examples, 1):
        cleaned_text = clean_text_conservative(text)
        text_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]
        
        result = "REAL" if prediction == 1 else "FAKE"
        print(f"{i}. {text[:50]}...")
        print(f"   Predicted: {result}")
        
        if prediction == 0:  # Correctly identified as fake
            fake_correct += 1
            print("   ✅ CORRECT")
        else:
            print("   ❌ WRONG - should be FAKE")
        print()
    
    print("\n🧪 Testing with OBVIOUSLY REAL examples:")
    print("="*60)
    
    real_correct = 0
    for i, text in enumerate(obvious_real_examples, 1):
        cleaned_text = clean_text_conservative(text)
        text_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]
        
        result = "REAL" if prediction == 1 else "FAKE"
        print(f"{i}. {text[:50]}...")
        print(f"   Predicted: {result}")
        
        if prediction == 1:  # Correctly identified as real
            real_correct += 1
            print("   ✅ CORRECT")
        else:
            print("   ❌ WRONG - should be REAL")
        print()
    
    print("="*60)
    print(f"📊 BALANCED MODEL RESULTS:")
    print(f"   Fake News Detection: {fake_correct}/{len(obvious_fake_examples)} correct ({fake_correct/len(obvious_fake_examples)*100:.1f}%)")
    print(f"   Real News Detection: {real_correct}/{len(obvious_real_examples)} correct ({real_correct/len(obvious_real_examples)*100:.1f}%)")
    print(f"   Overall Accuracy: {(fake_correct + real_correct)/(len(obvious_fake_examples) + len(obvious_real_examples))*100:.1f}%")
    
    if fake_correct >= len(obvious_fake_examples) * 0.7 and real_correct >= len(obvious_real_examples) * 0.7:
        print("   ✅ Model appears to be MUCH MORE BALANCED!")
    elif fake_correct < len(obvious_fake_examples) * 0.3:
        print("   🚨 Still biased towards REAL news")
    elif real_correct < len(obvious_real_examples) * 0.3:
        print("   🚨 Now biased towards FAKE news")
    else:
        print("   ⚠️ Moderate improvement but still needs work")

if __name__ == "__main__":
    test_balanced_model()
