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
    """Load the WELFake dataset with careful preprocessing"""
    # Load the new WELFake dataset
    dataset_path = "D:\\Kartik\\Learning\\ML\\Data\\WELFake_Dataset.csv"
    df = pd.read_csv(dataset_path)
    
    print(f"📊 Original WELFake dataset: {len(df)} articles")
    
    # Check the label distribution
    label_counts = df['label'].value_counts()
    print(f"📈 Label distribution: Real (1)={label_counts.get(1, 0)}, Fake (0)={label_counts.get(0, 0)}")
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Rename label column to class for consistency
    df = df.rename(columns={'label': 'class'})
    
    # Sample data more aggressively to ensure balanced training
    # Use a smaller, more balanced sample
    n_samples_per_class = 15000  # Reduced from 25k to 15k per class
    
    fake_samples = df[df['class'] == 0].sample(n=min(n_samples_per_class, len(df[df['class'] == 0])), random_state=42)
    real_samples = df[df['class'] == 1].sample(n=min(n_samples_per_class, len(df[df['class'] == 1])), random_state=42)
    
    df = pd.concat([fake_samples, real_samples], ignore_index=True)
    print(f"🎯 Balanced sample: {len(df)} articles ({len(fake_samples)} fake, {len(real_samples)} real)")
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def clean_text_less_aggressive(text):
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

def preprocess_dataframe(df):
    """Process dataframe with less aggressive cleaning"""
    # Handle missing values in title and text
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    
    # Combine title and text
    df["total_text"] = df["title"] + " " + df["text"]
    
    # Clean text with less aggressive approach
    df["total_text"] = df["total_text"].apply(clean_text_less_aggressive)
    
    # Be more lenient with text length requirements
    df = df[df["total_text"].str.len() > 10]  # Reduced from 20
    df = df[df["total_text"].str.len() < 15000]  # Increased from 10000
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=['total_text'])
    
    print(f"📝 After preprocessing: {len(df)} articles")
    print(f"🔍 Class distribution after preprocessing: {df['class'].value_counts().to_dict()}")
    
    # Check if we still have both classes
    class_counts = df['class'].value_counts()
    if len(class_counts) < 2:
        print("⚠️ WARNING: Only one class remains after preprocessing!")
        return None
    
    return df[["total_text", "class"]]

def create_better_features(df):
    """Create features with settings that preserve fake news indicators"""
    # TF-IDF settings that preserve more features
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Increased to capture more features
        min_df=2,              # Reduced minimum document frequency
        max_df=0.98,           # Increased maximum document frequency
        ngram_range=(1, 3),    # Include trigrams for better context
        strip_accents='unicode',
        lowercase=True,
        sublinear_tf=True,
        use_idf=True,
        stop_words=None        # Don't use built-in stop words, we handled this manually
    )
    
    X = vectorizer.fit_transform(df["total_text"])
    y = df["class"]
    
    print(f"📊 Feature matrix: {X.shape}")
    print(f"📈 Class distribution: Real={sum(y)}, Fake={len(y) - sum(y)}")
    
    return X, y, vectorizer

def train_balanced_model(X, y):
    """Train model with explicit focus on balance"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {len(y_train)} samples")
    print(f"   Real: {sum(y_train)}, Fake: {len(y_train) - sum(y_train)}")
    
    # Calculate class weights manually to ensure balance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"🎯 Class weights: {class_weight_dict}")
    
    # Try different models with explicit balancing (fast models only)
    models = {
        'SVM_Linear': LinearSVC(
            C=0.5, 
            class_weight=class_weight_dict,
            random_state=42, 
            max_iter=5000,
            dual=False  # More stable for this problem size
        ),
        'LogisticRegression': LogisticRegression(
            C=0.5,  # Reduced for regularization 
            class_weight=class_weight_dict, 
            random_state=42, 
            max_iter=2000,
            solver='liblinear',
            n_jobs=-1  # Use parallel processing
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,  # Use parallel processing
            max_depth=20  # Limit depth for speed
        )
    }
    
    best_model = None
    best_balanced_acc = 0
    best_name = ""
    results = {}
    
    print("🔬 Testing different models with balanced focus...")
    
    for name, model in models.items():
        # Cross-validation with balanced scoring
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_balanced = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'cv_balanced_acc': cv_balanced.mean(),
            'cv_f1': cv_f1.mean(),
            'test_balanced_acc': test_balanced_acc,
            'test_f1': test_f1,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"   {name}:")
        print(f"      CV Balanced Acc: {cv_balanced.mean():.4f} (+/- {cv_balanced.std()*2:.4f})")
        print(f"      Test Balanced Acc: {test_balanced_acc:.4f}, F1: {test_f1:.4f}")
        
        # Select best model based on balanced accuracy
        if test_balanced_acc > best_balanced_acc:
            best_balanced_acc = test_balanced_acc
            best_model = model
            best_name = name
    
    print(f"🏆 Best model: {best_name} with Balanced Accuracy: {best_balanced_acc:.4f}")
    
    return best_model, X_test, y_test, results[best_name]['predictions'], results

def evaluate_balance_focused(y_test, y_pred, model_name="Model"):
    """Evaluation focused on detecting bias"""
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n🎯 {model_name} Balance-Focused Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Per-class metrics
    precision, recall, f1_scores, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    print(f"\n📊 Per-Class Performance:")
    print(f"   FAKE News (0): Precision={precision[0]:.3f}, Recall={recall[0]:.3f}, F1={f1_scores[0]:.3f}")
    print(f"   REAL News (1): Precision={precision[1]:.3f}, Recall={recall[1]:.3f}, F1={f1_scores[1]:.3f}")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🔍 Confusion Matrix:")
    print(f"   [[{tn:4d} {fp:4d}]   <- Fake news")
    print(f"    [{fn:4d} {tp:4d}]]   <- Real news")
    print("      ^    ^")
    print("   Fake Real predictions")
    
    # Bias detection
    total_predictions = len(y_pred)
    fake_predicted = np.sum(y_pred == 0)
    real_predicted = np.sum(y_pred == 1)
    
    print(f"\n⚖️ Bias Analysis:")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Predicted as FAKE: {fake_predicted} ({fake_predicted/total_predictions*100:.1f}%)")
    print(f"   Predicted as REAL: {real_predicted} ({real_predicted/total_predictions*100:.1f}%)")
    
    if real_predicted / total_predictions > 0.8:
        print("   🚨 SEVERE BIAS towards REAL news!")
    elif real_predicted / total_predictions > 0.7:
        print("   ⚠️  MODERATE BIAS towards REAL news")
    elif fake_predicted / total_predictions > 0.8:
        print("   🚨 SEVERE BIAS towards FAKE news!")
    elif fake_predicted / total_predictions > 0.7:
        print("   ⚠️  MODERATE BIAS towards FAKE news")
    else:
        print("   ✅ Reasonably balanced predictions")
    
    return accuracy, balanced_acc, f1

def save_model(model, vectorizer):
    """Save the balanced model"""
    joblib.dump(model, "svm_model_balanced.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer_balanced.pkl")
    print("✅ Balanced model and vectorizer saved!")

def main():
    """Train a truly balanced fake news detector"""
    print("🚀 Training a BALANCED fake news detector (Version 2)!")
    print("📚 Loading WELFake data with careful balancing...")
    df = load_and_preprocess_data()
    
    print("🧹 Less aggressive text preprocessing...")
    df = preprocess_dataframe(df)
    
    if df is None:
        print("❌ Preprocessing failed. Exiting.")
        return
    
    print("🔢 Creating better features...")
    X, y, vectorizer = create_better_features(df)
    
    print("🤖 Training balanced models...")
    best_model, X_test, y_test, y_pred, all_results = train_balanced_model(X, y)
    
    print("📈 Balance-focused evaluation...")
    accuracy, balanced_acc, f1 = evaluate_balance_focused(y_test, y_pred, "Best Balanced Model")
    
    print("💾 Saving the balanced model...")
    save_model(best_model, vectorizer)
    
    print(f"\n🎉 Balanced training completed!")
    print(f"📊 Final Performance: Accuracy={accuracy:.4f}, Balanced Acc={balanced_acc:.4f}, F1={f1:.4f}")
    print(f"🎯 This model should be much more balanced and less biased!")
    print("🔥 Test it with: python test_balanced_model.py")

if __name__ == "__main__":
    main()
