"""
Optimized training script for fake news detection with reduced training time
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import re
from nltk.corpus import stopwords
import nltk
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_and_preprocess_data():
    """Load datasets with better sampling to reduce bias"""
    print("Starting data loading...")
    start_time = time.time()
    
    # Load datasets
    url_fake = "D:\\Kartik\\Learning\\ML\\Data\\Fake.csv"
    url_true = "D:\\Kartik\\Learning\\ML\\Data\\True.csv"
    df_fake = pd.read_csv(url_fake)
    df_true = pd.read_csv(url_true)
    
    print(f"📊 Original data: Fake={len(df_fake)}, Real={len(df_true)}")
    
    # Balance the dataset by taking equal samples - with smaller sample size for faster training
    min_samples = min(len(df_fake), len(df_true))
    sample_size = min(min_samples, 2000)  # Reduced sample size for faster training
    
    df_fake_sampled = df_fake.sample(n=sample_size, random_state=42)
    df_true_sampled = df_true.sample(n=sample_size, random_state=42)
    
    print(f"🎯 Balanced data: Fake={len(df_fake_sampled)}, Real={len(df_true_sampled)}")
    
    # Add class labels
    df_fake_sampled["class"] = 0  # Fake news
    df_true_sampled["class"] = 1  # Real news
    
    # Combine datasets
    df = pd.concat([df_fake_sampled, df_true_sampled], axis=0)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    
    return df

def clean_text_conservative(text):
    """More conservative text cleaning to preserve meaning"""
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

def preprocess_dataframe(df):
    """Process dataframe with better handling"""
    print("Starting text preprocessing...")
    start_time = time.time()
    
    # Combine title and text
    df["total_text"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    
    # Clean text
    df["total_text"] = df["total_text"].apply(clean_text_conservative)
    
    # Remove very short or very long texts
    df = df[(df["total_text"].str.len() > 30) & (df["total_text"].str.len() < 5000)]
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=['total_text'])
    
    print(f"📝 After preprocessing: {len(df)} articles")
    print(f"Text preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    return df[["total_text", "class"]]

def create_balanced_features(df):
    """Create features with better balance"""
    print("Starting feature creation...")
    start_time = time.time()
    
    # More balanced TF-IDF settings with fewer features for speed
    vectorizer = TfidfVectorizer(
        max_features=1500,     # Reduced for faster training
        min_df=3,             # Must appear in at least 3 documents
        max_df=0.95,          # Can appear in up to 95% of documents
        ngram_range=(1, 2),   # Unigrams and bigrams
        strip_accents='unicode',
        lowercase=True,
        sublinear_tf=True,    # Use sublinear TF scaling
        use_idf=True
    )
    
    X = vectorizer.fit_transform(df["total_text"])
    y = df["class"]
    
    print(f"📊 Feature matrix: {X.shape}")
    print(f"📈 Class distribution: Real={sum(y)}, Fake={len(y) - sum(y)}")
    print(f"Feature creation completed in {time.time() - start_time:.2f} seconds")
    
    return X, y, vectorizer

def train_models_fast(X, y):
    """Train models without cross-validation for faster execution"""
    print("Starting model training...")
    start_time = time.time()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'SVM_Linear': SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42, probability=True),
        'LogisticRegression': LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
    }
    
    best_model = None
    best_f1 = 0
    best_name = ""
    results = {}
    
    print("🔬 Testing different models (fast mode)...")
    
    for name, model in models.items():
        model_start = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Test the model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        test_f1 = f1_score(y_test, y_pred, average='macro')
        test_acc = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'test_f1': test_f1,
            'test_acc': test_acc,
            'model': model,
            'predictions': y_pred
        }
        
        model_time = time.time() - model_start
        print(f"   {name}: F1={test_f1:.4f}, Acc={test_acc:.4f} (in {model_time:.2f} seconds)")
        
        # Select best model based on F1 score
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = model
            best_name = name
    
    print(f"🏆 Best model: {best_name} with F1-score: {best_f1:.4f}")
    print(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    return best_model, X_test, y_test, results[best_name]['predictions'], results

def evaluate_model_detailed(y_test, y_pred, model_name="Model"):
    """Detailed evaluation focusing on balance"""
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n🎯 {model_name} Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake News', 'Real News']))
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🔍 Confusion Matrix:")
    print(f"   True Negatives (Fake → Fake): {tn}")
    print(f"   False Positives (Fake → Real): {fp}")
    print(f"   False Negatives (Real → Fake): {fn}")
    print(f"   True Positives (Real → Real): {tp}")
    
    # Calculate balance metrics
    fake_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    real_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fake_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    real_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n⚖️ Balance Analysis:")
    print(f"   Fake News - Precision: {fake_precision:.4f}, Recall: {fake_recall:.4f}")
    print(f"   Real News - Precision: {real_precision:.4f}, Recall: {real_recall:.4f}")
    
    # Check for bias
    bias_towards_fake = fn / (fn + tp) if (fn + tp) > 0 else 0
    bias_towards_real = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n🚨 Bias Check:")
    print(f"   Bias towards FAKE: {bias_towards_fake:.4f} (lower is better)")
    print(f"   Bias towards REAL: {bias_towards_real:.4f} (lower is better)")
    
    if bias_towards_fake > 0.3:
        print("   ⚠️ Model is biased towards predicting FAKE news!")
    elif bias_towards_real > 0.3:
        print("   ⚠️ Model is biased towards predicting REAL news!")
    else:
        print("   ✅ Model seems reasonably balanced!")
    
    return accuracy, f1

def save_model(model, vectorizer):
    """Save the best model"""
    joblib.dump(model, "svm_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("✅ Best model and vectorizer saved!")

def main():
    """Train a balanced fake news detector"""
    total_start_time = time.time()
    print("🚀 Training a BALANCED fake news detector! (FAST MODE)")
    print("===============================================")
    
    df = load_and_preprocess_data()
    df = preprocess_dataframe(df)
    X, y, vectorizer = create_balanced_features(df)
    
    best_model, X_test, y_test, y_pred, all_results = train_models_fast(X, y)
    accuracy, f1 = evaluate_model_detailed(y_test, y_pred, "Best Model")
    
    save_model(best_model, vectorizer)
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 Training completed in {total_time:.2f} seconds!")
    print(f"📊 Final Performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    print(f"🎯 This model should be balanced and train much faster!")
    print("🔥 Test it with: python test_model.py")

if __name__ == "__main__":
    main()
