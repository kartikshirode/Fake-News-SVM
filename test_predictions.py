import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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

def test_model_predictions():
    """Test the model with various examples"""
    print("🔍 Loading model and testing predictions...")
    
    # Load models
    try:
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Test with various examples
    test_cases = [
        ("BREAKING: Scientists discover aliens on Mars!", "Should be fake"),
        ("The Federal Reserve announced a 0.25% interest rate increase today following their monthly meeting.", "Should be real"),
        ("SHOCKING: This one trick doctors don't want you to know will cure cancer instantly!", "Should be fake"),
        ("President Biden met with European leaders to discuss trade agreements.", "Should be real"),
        ("UNBELIEVABLE: Local woman loses 50 pounds in 2 days with this miracle pill!", "Should be fake"),
        ("The stock market closed up 2% today amid positive economic indicators.", "Should be real"),
        ("EXPOSED: Government hiding the truth about flat earth theory!", "Should be fake"),
        ("Weather forecast shows heavy rain expected this weekend across the region.", "Should be real")
    ]
    
    print("\n🧪 Testing with sample cases:")
    print("="*80)
    
    predictions = []
    for text, expected in test_cases:
        # Clean the text
        cleaned_text = clean_text_conservative(text)
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        
        # Get confidence if available
        if hasattr(model, 'decision_function'):
            confidence = abs(model.decision_function(text_vector)[0])
        else:
            confidence = "N/A"
        
        predictions.append(prediction)
        
        result = "REAL" if prediction == 1 else "FAKE"
        print(f"Text: {text[:60]}...")
        print(f"Expected: {expected} | Predicted: {result} | Confidence: {confidence}")
        print("-" * 40)
    
    # Check if model is always predicting the same class
    unique_predictions = set(predictions)
    if len(unique_predictions) == 1:
        print("\n🚨 WARNING: Model is predicting the same class for all examples!")
        print(f"   All predictions are: {'REAL' if list(unique_predictions)[0] == 1 else 'FAKE'}")
        print("   This suggests the model is biased or broken.")
    else:
        print(f"\n✅ Model is making varied predictions: {len(unique_predictions)} different classes")
    
    return predictions

def test_on_original_data():
    """Test on a sample of the original dataset"""
    print("\n📊 Testing on original WELFake dataset...")
    
    try:
        # Load a small sample of the original data
        df = pd.read_csv("D:\\Kartik\\Learning\\ML\\Data\\WELFake_Dataset.csv")
        sample_df = df.sample(n=100, random_state=42)  # Test on 100 random samples
        
        # Load models
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        # Prepare the data
        sample_df["title"] = sample_df["title"].fillna("").astype(str)
        sample_df["text"] = sample_df["text"].fillna("").astype(str)
        sample_df["total_text"] = sample_df["title"] + " " + sample_df["text"]
        sample_df["total_text"] = sample_df["total_text"].apply(clean_text_conservative)
        
        # Make predictions
        X_test = vectorizer.transform(sample_df["total_text"])
        y_pred = model.predict(X_test)
        y_true = sample_df["label"]
        
        # Show results
        print(f"📈 Results on {len(sample_df)} test samples:")
        print(f"   True labels: Real={sum(y_true)}, Fake={len(y_true)-sum(y_true)}")
        print(f"   Predictions: Real={sum(y_pred)}, Fake={len(y_pred)-sum(y_pred)}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))
        
        # Confusion matrix
        print("\n🔍 Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"   [[{cm[0,0]} {cm[0,1]}]   <- Fake news")
        print(f"    [{cm[1,0]} {cm[1,1]}]]   <- Real news")
        print("     ^    ^")
        print("  Fake Real")
        print("  predictions")
        
    except Exception as e:
        print(f"❌ Error testing on original data: {e}")

if __name__ == "__main__":
    test_model_predictions()
    test_on_original_data()
