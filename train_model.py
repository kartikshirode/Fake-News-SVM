import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
import nltk
import joblib
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load datasets with better sampling to reduce bias"""
    # Load datasets
    url_fake = "D:\\Kartik\\Learning\\ML\\Data\\Fake.csv"
    url_true = "D:\\Kartik\\Learning\\ML\\Data\\True.csv"
    df_fake = pd.read_csv(url_fake)
    df_true = pd.read_csv(url_true)
    
    print(f"📊 Original data: Fake={len(df_fake)}, Real={len(df_true)}")
    
    # Balance the dataset by taking equal samples
    min_samples = min(len(df_fake), len(df_true))
    sample_size = min(min_samples, 15000)  # Cap at 15k each to prevent overfitting
    
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
    # Combine title and text
    df["total_text"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    
    # Clean text
    df["total_text"] = df["total_text"].apply(clean_text_conservative)
    
    # Remove very short or very long texts
    df = df[(df["total_text"].str.len() > 30) & (df["total_text"].str.len() < 5000)]
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=['total_text'])
    
    print(f"📝 After preprocessing: {len(df)} articles")
    
    return df[["total_text", "class"]]

def create_balanced_features(df):
    """Create features with better balance"""
    # More balanced TF-IDF settings
    vectorizer = TfidfVectorizer(
        max_features=2000,     # Moderate number of features
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
    
    return X, y, vectorizer

def train_multiple_models(X, y):
    """Try multiple models and pick the best balanced one"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'SVM_Linear': LinearSVC(C=1.0, class_weight='balanced',random_state=42, max_iter=5000),
        'LogisticRegression': LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_f1 = 0
    best_name = ""
    results = {}
    
    print("🔬 Testing different models...")
    
    for name, model in models.items():
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_f1 = f1_score(y_test, y_pred, average='macro')
        test_acc = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_acc_mean': cv_acc.mean(),
            'test_f1': test_f1,
            'test_acc': test_acc,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"   {name}:")
        print(f"      CV F1: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
        print(f"      Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}")
        
        # Select best model based on F1 score (better for balanced classification)
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = model
            best_name = name
    
    print(f"🏆 Best model: {best_name} with F1-score: {best_f1:.4f}")
    
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
    print("🚀 Training a BALANCED fake news detector!")
    print("📚 Loading data with balanced sampling...")
    df = load_and_preprocess_data()
    
    print("🧹 Conservative text preprocessing...")
    df = preprocess_dataframe(df)
    # Debug: show class distribution after preprocessing (to detect imbalance causing single-class predictions)
    class_counts = df['class'].value_counts().to_dict()
    print(f"🔎 Class distribution AFTER preprocessing: {class_counts}")
    if len(class_counts) < 2:
        print("⚠️ WARNING: Only one class remains after preprocessing. Predictions will all be the same. Adjust cleaning or data sampling.")
        print("   Suggested quick fixes:\n"
              "   - Reduce aggressiveness of stopword removal\n"
              "   - Skip removing very short texts threshold or lower length cutoff\n"
              "   - Ensure both Fake and Real CSVs loaded correctly")
    
    print("🔢 Creating balanced features...")
    X, y, vectorizer = create_balanced_features(df)
    
    print("🤖 Training multiple models to find the best one...")
    best_model, X_test, y_test, y_pred, all_results = train_multiple_models(X, y)
    
    print("📈 Detailed evaluation...")
    accuracy, f1 = evaluate_model_detailed(y_test, y_pred, "Best Model")
    
    print("💾 Saving the best model...")
    save_model(best_model, vectorizer)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final Performance: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    print(f"🎯 This model should be much more balanced!")
    print("🔥 Test it with: python test_model.py")

if __name__ == "__main__":
    main()
